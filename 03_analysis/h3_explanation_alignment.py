import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest, chi2_contingency
from statsmodels.stats.proportion import proportion_confint
import statsmodels.formula.api as smf

# paths
base_dir = Path(__file__).parent
log_dir = base_dir / "logs"
fig_dir = base_dir / "figures"
log_dir.mkdir(exist_ok=True)
fig_dir.mkdir(exist_ok=True)

# logger
logger = logging.getLogger("h3")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(log_dir / "h3_explanation_alignment.log", mode="w", encoding="utf-8")
ch = logging.StreamHandler()
for h in (fh, ch):
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
def log(m): logger.info(m)

FEATURE_TERMS = {
    "company_size": ["company size", "company of", "employees", "size (", "large company",
                     "small company", "large enterprise", "mid-size", "mid-sized",
                     "medium to large", "small (", "large ("],
    "website_dwell_time": ["dwell time", "dwell_time", "time on site", "sec)", "seconds",
                           " sec", "min)", "minute", " min", "time spent"],
    "demo_requests": ["demo request", "demo requests", "demo_requests", "demo", "no demo"],
    "email_response": ["email response", "email_response", "email", "response to", "reply",
                       "not interested", "looks interesting", "send more info",
                       "schedule a meeting", "meeting next week", "relevant to",
                       "very promising", "negative response", "disinterest", "responded"],
    "industry": ["industry", "finance", "public service", "software", "manufacturing",
                 "healthcare", "energy", "retail", "sector"],
    "region": ["region", "europe", "north america", "south america", "africa",
               "middle east", "asia", "geograph", "based in"],
}

# ---------- load coded explanations ----------
coding = pd.read_csv(base_dir / "post_processing" / "llm_lead_intent_results_explanations_coded.csv")
coding["aligned"] = (coding["final_code"] > 0).astype(int)

low_text = coding["reasoning"].astype(str).str.lower()
for f, terms in FEATURE_TERMS.items():
    coding[f"kw_{f}"] = low_text.apply(lambda t: int(any(k in t for k in terms)))

agg = {"aligned": ("aligned", lambda s: int(s.mean() >= 0.5)),
       "align_share": ("aligned", "mean")}
for f in FEATURE_TERMS:
    agg[f"kw_{f}"] = (f"kw_{f}", lambda s: int(s.mean() >= 0.5))
case = (coding.groupby(["profile_id", "model", "perturbed_feature"])
        .agg(**agg).reset_index())

# ---------- determine which perturbation changed the classification (for Variant 1) ----------
scores = pd.read_csv(base_dir / "post_processing" / "llm_lead_intent_results_clean.csv")
scores["label"] = scores["intent"].str.strip().map({"Low Intent": 0, "High Intent": 1})
scores["base_profile"] = scores["profile_id"].str.split("_").str[0]
scores["perturbation"] = scores["profile_id"].str.split("_", n=1).str[1]
maj = (scores.groupby(["profile_id", "model", "base_profile", "perturbation"])["label"]
       .agg(lambda s: int(s.mean() >= 0.5)).reset_index().rename(columns={"label": "maj_label"}))
base_lab = (maj[maj["perturbation"] == "BASE"][["base_profile", "model", "maj_label"]]
            .rename(columns={"maj_label": "base_label"}))
pert = maj[maj["perturbation"] != "BASE"].merge(base_lab, on=["base_profile", "model"], how="left")
pert["score_changed"] = (pert["maj_label"] != pert["base_label"]).astype(int)
case = case.merge(pert[["profile_id", "model", "score_changed"]], on=["profile_id", "model"], how="left")
case["score_changed"] = case["score_changed"].fillna(0).astype(int)

# ---------- STEP 0: quantify the causal case base (motivation for Variant 2) ----------
n_total = len(case)
n_causal = int(case["score_changed"].sum())
n_stable = n_total - n_causal
log("H3 Explanation Alignment")
log("========================")
log("Rationale: H3 targets causal cases (score changed). We first quantify how")
log("many such cases exist to justify the complementary all-cases analysis.")
log(f"\ntotal perturbation cases: {n_total}")
log(f"score-CHANGING cases (Variant 1 base): {n_causal} ({n_causal / n_total:.1%})")
log(f"score-stable cases: {n_stable} ({n_stable / n_total:.1%})")
if n_causal < 30:
    log(f">> Only {n_causal} causal cases: statistical power for Variant 1 is limited,")
    log(">> so Variant 2 (all perturbation cases) is reported as the primary evidence.")

def report(sub, title):
    n = len(sub)
    if n == 0:
        log(f"\n{title}: no cases"); return
    a = int(sub["aligned"].sum())
    r = a / n
    lo, hi = proportion_confint(a, n, alpha=0.05, method="wilson")
    log(f"\n{title}")
    log(f"  aligned: {a}/{n} = {r:.3f}   Wilson 95% CI [{lo:.3f}, {hi:.3f}]   NOT aligned {1 - r:.3f}")
    if n >= 5:
        bt = binomtest(a, n, 0.5, alternative="two-sided")
        log(f"  binomial test vs 0.5: p={bt.pvalue:.4g}")

# ---------- VARIANT 1: causal cases (the actual H3) ----------
report(case[case["score_changed"] == 1], "Variant 1 (H3, causal cases): perturbed feature cited")

# ---------- VARIANT 2: all perturbation cases (complementary) ----------
report(case, "Variant 2 (complementary, all cases): perturbed feature cited")

# empirical incidental baseline over the non-perturbed features (all cases)
incidental = [r[f"kw_{f}"] for _, r in case.iterrows()
              for f in FEATURE_TERMS if f != r["perturbed_feature"]]
p0 = float(np.mean(incidental))
a_all, n_all = int(case["aligned"].sum()), len(case)
bt0 = binomtest(a_all, n_all, p0, alternative="two-sided")
log(f"\nincidental baseline p0={p0:.3f}; Variant 2 binomial vs p0: p={bt0.pvalue:.4g}")

# ---------- feature-type analyses on Variant 2 (enough power) ----------
log("\n--- Alignment per perturbed feature (Variant 2) ---")
feat = (case.groupby("perturbed_feature")["aligned"]
        .agg(["mean", "sum", "count"]).sort_values("mean"))
log(feat.to_string())

cont = pd.crosstab(case["perturbed_feature"], case["aligned"])
chi2, p_chi, dof, _ = chi2_contingency(cont)
V = np.sqrt(chi2 / (cont.values.sum() * (min(cont.shape) - 1)))
log(f"\nchi-square feature vs alignment: chi2={chi2:.2f}, dof={dof}, p={p_chi:.4g}, Cramer's V={V:.3f}")

m = smf.logit("aligned ~ C(perturbed_feature)", data=case).fit(disp=False)
log("\nlogistic regression aligned ~ C(perturbed_feature):")
log(m.summary2().tables[1].to_string())

# ---------- visualizations ----------
sns.set_theme(style="whitegrid")

# Plot 1: case base — how many cases changed the score
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(["score stable", "score changed\n(H3 base)"], [n_stable, n_causal],
       color=["#9ecae1", "#08519c"])
for i, v in enumerate([n_stable, n_causal]):
    ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
ax.set_ylabel("Number of perturbation cases")
ax.set_title("H3: Few cases actually change the score (motivates Variant 2)")
fig.tight_layout()
fig.savefig(fig_dir / "h3_case_base.png", dpi=200)

# Plot 2: alignment Variant 1 vs Variant 2 with Wilson CIs
fig, ax = plt.subplots(figsize=(6, 5))
groups, rates, elo, ehi = [], [], [], []
for label, sub in [("V1 causal", case[case["score_changed"] == 1]),
                   ("V2 all", case)]:
    if len(sub) == 0:
        continue
    a, nn = int(sub["aligned"].sum()), len(sub)
    r = a / nn
    l, h = proportion_confint(a, nn, method="wilson")
    groups.append(f"{label}\n(n={nn})"); rates.append(r)
    elo.append(r - l); ehi.append(h - r)
ax.bar(groups, rates, yerr=[elo, ehi], capsize=6, color=["#08519c", "#2c7fb8"])
for i, r in enumerate(rates):
    ax.text(i, r + 0.03, f"{r:.2f}", ha="center", fontweight="bold")
ax.axhline(1.0, color="black", linestyle=":", label="perfect alignment")
ax.axhline(p0, color="#de2d26", linestyle="--", label=f"incidental baseline ({p0:.2f})")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Citation rate of perturbed feature")
ax.set_title("H3: Explanation alignment, causal vs all cases")
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "h3_alignment_v1_v2.png", dpi=200)

# Plot 3: alignment per feature (Variant 2) with Wilson CIs
fig, ax = plt.subplots(figsize=(7, 5))
fp = feat.reset_index()
elo, ehi = [], []
for _, r in fp.iterrows():
    l, h = proportion_confint(int(r["sum"]), int(r["count"]), method="wilson")
    elo.append(r["mean"] - l); ehi.append(h - r["mean"])
ax.barh(fp["perturbed_feature"], fp["mean"], xerr=[elo, ehi], capsize=4, color="#2c7fb8")
ax.axvline(p0, color="#de2d26", linestyle="--", label=f"incidental baseline ({p0:.2f})")
ax.set_xlim(0, 1.05)
ax.set_title("H3: Alignment rate per perturbed feature (all cases)")
ax.set_xlabel("Share of cases the feature is cited")
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "h3_alignment_per_feature.png", dpi=200)