import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from statsmodels.stats.proportion import proportion_confint

# paths
base_dir = Path(__file__).parent
log_dir = base_dir / "logs"
fig_dir = base_dir / "figures"
log_dir.mkdir(exist_ok=True)
fig_dir.mkdir(exist_ok=True)

# logger writes to file and console
logger = logging.getLogger("h2a")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(log_dir / "h2a_score_stability.log", mode="w", encoding="utf-8")
ch = logging.StreamHandler()
fmt = logging.Formatter("%(message)s")
fh.setFormatter(fmt)
ch.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)

def log(msg):
    # print and log at once
    logger.info(msg)

# load data
df = pd.read_csv(base_dir / "post_processing" / "llm_lead_intent_results_clean.csv")

# map intent to binary label
label_map = {"Low Intent": 0, "High Intent": 1}
df["label"] = df["intent"].str.strip().map(label_map)

# a case is one profile evaluated by one model, repeated over runs
df["case"] = df["profile_id"] + " | " + df["model"]

# ---------- per case consistency ----------
# majority share and whether all runs agree
def case_stats(g):
    n = len(g)
    counts = g["label"].value_counts()
    majority = counts.max()
    return pd.Series({
        "n_runs": n,
        "n_high": int((g["label"] == 1).sum()),
        "n_low": int((g["label"] == 0).sum()),
        "majority_share": majority / n,
        "consistent": int(majority == n),
    })

per_case = df.groupby(["case", "model", "profile_id"]).apply(case_stats).reset_index()
per_case["n_runs"] = per_case["n_runs"].astype(int)
per_case["consistent"] = per_case["consistent"].astype(int)

n_cases = len(per_case)
n_inconsistent = int((per_case["consistent"] == 0).sum())
inconsistent_rate = n_inconsistent / n_cases

# wilson interval for the inconsistency rate
ci_low, ci_high = proportion_confint(n_inconsistent, n_cases, alpha=0.05, method="wilson")

log("H2a Score Stability")
log("===================")
log(f"cases total: {n_cases}")
log(f"inconsistent cases: {n_inconsistent}")
log(f"inconsistency rate: {inconsistent_rate:.3f}  Wilson 95% CI [{ci_low:.3f}, {ci_high:.3f}]")
log(f"mean majority share: {per_case['majority_share'].mean():.3f}")

# ---------- Fleiss kappa over all cases ----------
# runs act as raters, cases are subjects, two categories
ratings_matrix, _ = aggregate_raters(
    df.pivot_table(index="case", columns="run", values="label").values
)
kappa = fleiss_kappa(ratings_matrix, method="fleiss")
log(f"\nFleiss kappa (runs as raters): {kappa:.3f}")

# ---------- Krippendorff alpha nominal ----------
# small nominal implementation to avoid extra dependency
def krippendorff_alpha_nominal(pivot):
    # pivot rows are subjects, columns are raters, values are labels or nan
    values = pivot.values.astype(float)
    units = []
    for row in values:
        row = row[~np.isnan(row)]
        if len(row) >= 2:
            units.append(row)
    # observed disagreement
    num_o = 0.0
    den_o = 0.0
    # coincidence counts
    cats = np.unique(np.concatenate(units))
    coinc = {c: {d: 0.0 for d in cats} for c in cats}
    for row in units:
        m = len(row)
        for i in range(m):
            for j in range(m):
                if i != j:
                    coinc[row[i]][row[j]] += 1.0 / (m - 1)
    n_total = sum(sum(coinc[c].values()) for c in cats)
    # expected disagreement uses marginal totals
    marg = {c: sum(coinc[c].values()) for c in cats}
    Do = 0.0
    for c in cats:
        for d in cats:
            if c != d:
                Do += coinc[c][d]
    Do = Do / n_total
    De = 0.0
    for c in cats:
        for d in cats:
            if c != d:
                De += marg[c] * marg[d]
    De = De / (n_total * (n_total - 1))
    return 1 - Do / De if De > 0 else 1.0

pivot_labels = df.pivot_table(index="case", columns="run", values="label")
alpha = krippendorff_alpha_nominal(pivot_labels)
log(f"Krippendorff alpha (nominal): {alpha:.3f}")

# ---------- bootstrap CI for kappa and alpha ----------
# resample cases with replacement to get confidence intervals
rng = np.random.default_rng(42)
n_boot = 2000
cases = pivot_labels.index.to_numpy()
boot_kappa = []
boot_alpha = []
for _ in range(n_boot):
    sample = rng.choice(cases, size=len(cases), replace=True)
    sub = pivot_labels.loc[sample]
    try:
        rm, _ = aggregate_raters(sub.values)
        boot_kappa.append(fleiss_kappa(rm, method="fleiss"))
    except Exception:
        pass
    boot_alpha.append(krippendorff_alpha_nominal(sub))

k_lo, k_hi = np.percentile(boot_kappa, [2.5, 97.5])
a_lo, a_hi = np.percentile(boot_alpha, [2.5, 97.5])
log(f"Fleiss kappa 95% CI [{k_lo:.3f}, {k_hi:.3f}]")
log(f"Krippendorff alpha 95% CI [{a_lo:.3f}, {a_hi:.3f}]")

# ---------- agreement per model ----------
# same metrics computed within each model
log("\nAgreement per model")
model_rows = []
for model, gm in df.groupby("model"):
    piv = gm.pivot_table(index="case", columns="run", values="label")
    rm, _ = aggregate_raters(piv.values)
    km = fleiss_kappa(rm, method="fleiss")
    am = krippendorff_alpha_nominal(piv)
    inc = int((piv.apply(lambda r: r.dropna().nunique() > 1, axis=1)).sum())
    model_rows.append({"model": model, "kappa": km, "alpha": am,
                       "n_cases": len(piv), "n_inconsistent": inc})
    log(f"  {model}: kappa={km:.3f}, alpha={am:.3f}, inconsistent={inc}/{len(piv)}")
model_df = pd.DataFrame(model_rows)

# Visualizations
sns.set_theme(style="whitegrid")

# Plot 1: share of consistent vs inconsistent cases
fig, ax = plt.subplots(figsize=(6, 5))
vals = [n_cases - n_inconsistent, n_inconsistent]
ax.bar(["consistent", "inconsistent"], vals, color=["#31a354", "#de2d26"])
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontweight="bold")
ax.set_title("H2a: Consistent vs inconsistent cases (5 runs each)")
ax.set_ylabel("Number of cases")
fig.tight_layout()
fig.savefig(fig_dir / "h2a_consistency_counts.png", dpi=200)

# Plot 2: distribution of majority share
fig, ax = plt.subplots(figsize=(6, 5))
sns.histplot(per_case["majority_share"], bins=[0.5, 0.7, 0.9, 1.01], ax=ax, color="#2c7fb8")
ax.set_title("H2a: Distribution of within-case majority share")
ax.set_xlabel("Majority share across 5 runs")
ax.set_ylabel("Number of cases")
fig.tight_layout()
fig.savefig(fig_dir / "h2a_majority_share_hist.png", dpi=200)

# Plot 3: agreement metrics per model with reference line at perfect agreement
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(model_df))
w = 0.35
ax.bar(x - w / 2, model_df["kappa"], w, label="Fleiss kappa", color="#2c7fb8")
ax.bar(x + w / 2, model_df["alpha"], w, label="Krippendorff alpha", color="#de2d26")
ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="perfect agreement")
ax.set_xticks(x)
ax.set_xticklabels(model_df["model"], rotation=20, ha="right")
ax.set_ylabel("Agreement")
ax.set_title("H2a: Inter-run agreement per model")
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "h2a_agreement_per_model.png", dpi=200)

# Plot 4: heatmap of inconsistent cases by profile and model
piv_inc = per_case.pivot_table(index="profile_id", columns="model",
                               values="consistent", aggfunc="min")
fig, ax = plt.subplots(figsize=(8, 16))
sns.heatmap(piv_inc, cmap="RdYlGn", yticklabels=True,
            cbar_kws={"label": "1 = consistent, 0 = inconsistent"},
            linewidths=0.5, ax=ax)
ax.tick_params(axis="y", labelsize=7)
ax.set_title("H2a: Consistency by profile and model")
fig.tight_layout()
fig.savefig(fig_dir / "h2a_consistency_heatmap.png", dpi=200)