import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# load data
base_dir = Path(__file__).parent
df = pd.read_csv(base_dir / "post_processing" / "llm_lead_intent_results_clean.csv")

# logging
log_dir = base_dir / "logs"
log_dir.mkdir(exist_ok=True)
log_path = log_dir / f"h1_feature_sensitivity.log"

(base_dir / "figures").mkdir(exist_ok=True)

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

_log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, _log_file)

# map intent classes to binary score
score_map = {"Low Intent": 0, "High Intent": 1}
df["score"] = df["intent"].str.strip().map(score_map)

# base profile letter, e.g. A_company_size_high to A
df["base_profile"] = df["profile_id"].str.split("_").str[0]

# perturbation part, e.g. company_size_high or BASE
df["perturbation"] = df["profile_id"].str.split("_", n=1).str[1]

# which feature was perturbed
def get_feature(pert):
    for f in ["company_size", "website_dwell_time", "demo_requests",
              "email_response", "industry", "region"]:
        if pert.startswith(f):
            return f
    return "BASE"
df["feature"] = df["perturbation"].apply(get_feature)

# direction of perturbation
def get_direction(pert):
    for d in ["non-icp", "high", "low", "icp"]:
        if pert.endswith(d):
            return d
    return "base"
df["direction"] = df["perturbation"].apply(get_direction)

# classify feature type
behavioral = ["website_dwell_time", "demo_requests", "email_response"]
firmographic = ["company_size", "industry", "region"]
def feature_type(f):
    if f in behavioral:
        return "behavioral"
    if f in firmographic:
        return "firmographic"
    return "base"
df["feature_type"] = df["feature"].apply(feature_type)

# baseline score per base profile, model and run
base = df[df["feature"] == "BASE"][["base_profile", "model", "run", "score"]]
base = base.rename(columns={"score": "base_score"})

# merge baseline back to perturbed rows
pert = df[df["feature"] != "BASE"].merge(
    base, on=["base_profile", "model", "run"], how="left")

# delta score relative to baseline
pert["delta"] = pert["score"] - pert["base_score"]

# absolute change magnitude
pert["abs_delta"] = pert["delta"].abs()

# ---------- Level 1: does a perturbation change the score at all ----------
# wilcoxon per feature comparing perturbed vs base scores
print("Level 1: Wilcoxon signed-rank per feature")
for f in behavioral + firmographic:
    sub = pert[pert["feature"] == f]
    try:
        stat, p = wilcoxon(sub["score"], sub["base_score"])
        print(f"  {f}: n={len(sub)}, p={p:.4f}")
    except ValueError:
        print(f"  {f}: no variation, test not applicable")

# ---------- Level 2: H1 main test, mixed-effects model ----------
# fixed effect feature_type, random intercept per base profile
print("\nLevel 2: Linear mixed-effects model")
model_lmm = smf.mixedlm("abs_delta ~ C(feature_type)", pert, groups=pert["base_profile"])
result = model_lmm.fit()
print(result.summary())

# ---------- Level 2: H1 supporting test, Mann-Whitney-U ----------
# compare abs delta between behavioral and firmographic features
beh = pert[pert["feature_type"] == "behavioral"]["abs_delta"]
firm = pert[pert["feature_type"] == "firmographic"]["abs_delta"]

# one-sided test: behavioral greater than firmographic
u, p = mannwhitneyu(beh, firm, alternative="greater")
print("\nLevel 2: Mann-Whitney-U (behavioral greater than firmographic)")
print(f"  n_behavioral={len(beh)}, n_firmographic={len(firm)}")
print(f"  U={u:.1f}, p={p:.4f}")
print(f"  median behavioral={beh.median()}, median firmographic={firm.median()}")

# effect size rank-biserial correlation
n1, n2 = len(beh), len(firm)
r_rb = 1 - (2 * u) / (n1 * n2)
print(f"  rank-biserial effect size={abs(r_rb):.3f}")

# ---------- descriptive summary ----------
print("\nDescriptive: mean abs delta per feature")
print(pert.groupby(["feature_type", "feature"])["abs_delta"].agg(["mean", "std", "count"]))

# Visualizations
sns.set_theme(style="whitegrid")
palette = {"behavioral": "#2c7fb8", "firmographic": "#de2d26"}
# ---------- Plot 0: natural intent distribution per base profile ----------
# use only BASE rows to see how each profile is rated without perturbation
base_rows = df[df["feature"] == "BASE"].copy()
base_rows["intent_label"] = base_rows["intent"].str.strip()
# counts of high and low intent per profile
counts = (base_rows.groupby(["base_profile", "intent_label"])
          .size().reset_index(name="n"))
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=counts, x="base_profile", y="n", hue="intent_label",
            palette={"High Intent": "#31a354", "Low Intent": "#de2d26"}, ax=ax)
ax.set_title("Natural intent distribution per base profile (unperturbed)")
ax.set_xlabel("Base profile")
ax.set_ylabel("Number of runs")
ax.legend(title="Intent")
fig.tight_layout()
fig.savefig(base_dir / "figures" / "h1_base_intent_distribution.png", dpi=200)
# ---------- Plot 1 (binary): flip rate by feature type with Wilson CI ----------
# aggregate successes and totals per feature type
agg = (pert.groupby("feature_type")["abs_delta"]
       .agg(successes="sum", total="count").reset_index())
agg["rate"] = agg["successes"] / agg["total"]
# wilson interval for each proportion
low, high = proportion_confint(agg["successes"], agg["total"],
                               alpha=0.05, method="wilson")
agg["err_low"] = agg["rate"] - low
agg["err_high"] = high - agg["rate"]
fig, ax = plt.subplots(figsize=(6, 5))
colors = [palette[ft] for ft in agg["feature_type"]]
ax.bar(agg["feature_type"], agg["rate"], color=colors,
       yerr=[agg["err_low"], agg["err_high"]], capsize=8)
# annotate bars with the flip rate
for i, row in agg.iterrows():
    ax.text(i, row["rate"] + row["err_high"] + 0.02, f"{row['rate']:.1%}",
            ha="center", va="bottom", fontweight="bold")
ax.set_title("H1: Class flip rate by feature type (Wilson 95% CI)")
ax.set_xlabel("Feature type")
ax.set_ylabel("Flip rate (share of runs that changed the class)")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(base_dir / "figures" / "h1_flip_rate.png", dpi=200)
# ---------- Plot 2: mean flip rate per feature ----------
# mean abs delta per single feature, colored by type
summary = (pert.groupby(["feature_type", "feature"])["abs_delta"]
           .mean().reset_index().sort_values("abs_delta"))
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=summary, x="abs_delta", y="feature",
            hue="feature_type", palette=palette, dodge=False, ax=ax)
ax.set_title("Mean score change per feature")
ax.set_xlabel("Mean absolute score change")
ax.set_ylabel("Feature")
ax.legend(title="Feature type")
fig.tight_layout()
fig.savefig(base_dir / "figures" / "h1_mean_change_per_feature.png", dpi=200)
# ---------- Plot 3: direction of change per feature (only used directions) ----------
# mean signed delta, observed feature and direction combinations only
dir_summary = (pert.groupby(["feature", "direction"])["delta"]
               .mean().reset_index())
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=dir_summary, x="feature", y="delta", hue="direction", ax=ax)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Direction of score change per feature and perturbation")
ax.set_xlabel("Feature")
ax.set_ylabel("Mean signed score change")
ax.tick_params(axis="x", rotation=30)
ax.legend(title="Direction")
fig.tight_layout()
fig.savefig(base_dir / "figures" / "h1_direction_per_feature.png", dpi=200)
# ---------- Plot 4: heatmap profile by feature ----------
# mean abs delta across base profiles and features
pivot = pert.pivot_table(index="base_profile", columns="feature",
                         values="abs_delta", aggfunc="mean")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
            cbar_kws={"label": "Mean absolute score change"}, ax=ax)
ax.set_title("Score change by profile and feature")
ax.set_xlabel("Feature")
ax.set_ylabel("Base profile")
fig.tight_layout()
fig.savefig(base_dir / "figures" / "h1_heatmap_profile_feature.png", dpi=200)
# show all figures
plt.show()

# restore stdout and close log file
sys.stdout = sys.__stdout__
_log_file.close()