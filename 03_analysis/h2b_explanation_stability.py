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
logger = logging.getLogger("h2b")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(log_dir / "h2b_explanation_stability.log", mode="w", encoding="utf-8")
ch = logging.StreamHandler()
for h in (fh, ch):
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)
def log(m): logger.info(m)

# load coded data
df = pd.read_csv(base_dir / "post_processing" / "llm_lead_intent_results_explanations_coded.csv")

# a case is one profile evaluated by one model, repeated over runs
df["case"] = df["profile_id"] + " | " + df["model"]

# also a coarse mention flag: is the perturbed feature mentioned at all
df["mentioned"] = (df["final_code"] > 0).astype(int)

# ---------- per case consistency of the full code ----------
# majority share and whether all runs share the same code
def case_stats(g):
    n = len(g)
    counts = g["final_code"].value_counts()
    majority = counts.max()
    return pd.Series({
        "n_runs": n,
        "majority_code": counts.idxmax(),
        "majority_share": majority / n,
        "consistent": int(majority == n),
    })

per_case = df.groupby(["case", "model", "profile_id", "perturbed_feature"]).apply(case_stats).reset_index()
per_case["n_runs"] = per_case["n_runs"].astype(int)
per_case["consistent"] = per_case["consistent"].astype(int)

n_cases = len(per_case)
n_inconsistent = int((per_case["consistent"] == 0).sum())
inconsistent_rate = n_inconsistent / n_cases

# wilson interval for the inconsistency rate
ci_low, ci_high = proportion_confint(n_inconsistent, n_cases, alpha=0.05, method="wilson")

log("H2b Explanation Stability")
log("=========================")
log(f"cases total: {n_cases}")
log(f"inconsistent cases: {n_inconsistent}")
log(f"inconsistency rate: {inconsistent_rate:.3f}  Wilson 95% CI [{ci_low:.3f}, {ci_high:.3f}]")
log(f"mean majority share: {per_case['majority_share'].mean():.3f}")

# ---------- overlap rates ----------
# code overlap: share of runs equal to the case majority code, averaged over cases
# this is the pairwise style overlap used for referenced features
def pairwise_overlap(codes):
    # fraction of run pairs that share the same value
    vals = codes.to_numpy()
    m = len(vals)
    agree = 0
    total = 0
    for i in range(m):
        for j in range(i + 1, m):
            total += 1
            if vals[i] == vals[j]:
                agree += 1
    return agree / total if total else np.nan

overlap_code = df.groupby("case")["final_code"].apply(pairwise_overlap).mean()
overlap_mention = df.groupby("case")["mentioned"].apply(pairwise_overlap).mean()
log(f"\nmean pairwise code overlap (0-3): {overlap_code:.3f}")
log(f"mean pairwise mention overlap (0 vs >0): {overlap_mention:.3f}")

# ---------- Fleiss kappa over all cases ----------
# runs act as raters, cases are subjects, four code categories
pivot_codes = df.pivot_table(index="case", columns="run", values="final_code")
ratings_matrix, _ = aggregate_raters(pivot_codes.values)
kappa = fleiss_kappa(ratings_matrix, method="fleiss")
log(f"\nFleiss kappa (runs as raters, codes 0-3): {kappa:.3f}")

# ---------- Krippendorff alpha ----------
# ordinal metric fits the ordered coding scheme 0<1<2<3
def krippendorff_alpha(pivot, level="ordinal"):
    values = pivot.values.astype(float)
    units = [row[~np.isnan(row)] for row in values]
    units = [u for u in units if len(u) >= 2]
    cats = np.unique(np.concatenate(units))
    # coincidence matrix
    coinc = {c: {d: 0.0 for d in cats} for c in cats}
    for row in units:
        m = len(row)
        for i in range(m):
            for j in range(m):
                if i != j:
                    coinc[row[i]][row[j]] += 1.0 / (m - 1)
    n_total = sum(sum(coinc[c].values()) for c in cats)
    marg = {c: sum(coinc[c].values()) for c in cats}

    def delta(c, d):
        if c == d:
            return 0.0
        if level == "nominal":
            return 0.0 if c == d else 1.0
        # ordinal distance based on cumulative marginals
        lo, hi = sorted([c, d])
        s = marg[lo] / 2 + marg[hi] / 2
        for g in cats:
            if lo < g < hi:
                s += marg[g]
        return s ** 2

    Do = sum(coinc[c][d] * delta(c, d) for c in cats for d in cats)
    De = sum(marg[c] * marg[d] * delta(c, d) for c in cats for d in cats) / (n_total - 1)
    return 1 - Do / De if De > 0 else 1.0

alpha = krippendorff_alpha(pivot_codes, level="ordinal")
log(f"Krippendorff alpha (ordinal): {alpha:.3f}")

# ---------- bootstrap CI for kappa and alpha ----------
# resample cases with replacement to get confidence intervals
rng = np.random.default_rng(42)
n_boot = 2000
cases = pivot_codes.index.to_numpy()
boot_kappa, boot_alpha = [], []
for _ in range(n_boot):
    sample = rng.choice(cases, size=len(cases), replace=True)
    sub = pivot_codes.loc[sample]
    try:
        rm, _ = aggregate_raters(sub.values)
        boot_kappa.append(fleiss_kappa(rm, method="fleiss"))
    except Exception:
        pass
    boot_alpha.append(krippendorff_alpha(sub, level="ordinal"))

k_lo, k_hi = np.percentile(boot_kappa, [2.5, 97.5])
a_lo, a_hi = np.percentile(boot_alpha, [2.5, 97.5])
log(f"Fleiss kappa 95% CI [{k_lo:.3f}, {k_hi:.3f}]")
log(f"Krippendorff alpha 95% CI [{a_lo:.3f}, {a_hi:.3f}]")

# ---------- agreement per model ----------
# same metrics computed within each model
log("\nAgreement per model")
model_rows = []
for model, gm in df.groupby("model"):
    piv = gm.pivot_table(index="case", columns="run", values="final_code")
    rm, _ = aggregate_raters(piv.values)
    km = fleiss_kappa(rm, method="fleiss")
    am = krippendorff_alpha(piv, level="ordinal")
    inc = int((piv.apply(lambda r: r.dropna().nunique() > 1, axis=1)).sum())
    ov = gm.groupby("case")["final_code"].apply(pairwise_overlap).mean()
    model_rows.append({"model": model, "kappa": km, "alpha": am,
                       "overlap": ov, "n_cases": len(piv), "n_inconsistent": inc})
    log(f"  {model}: kappa={km:.3f}, alpha={am:.3f}, overlap={ov:.3f}, "
        f"inconsistent={inc}/{len(piv)}")
model_df = pd.DataFrame(model_rows)

# ---------- consistency per feature ----------
# does stability depend on which feature was perturbed
log("\nConsistency per perturbed feature")
feat = (per_case.groupby("perturbed_feature")["consistent"]
        .agg(["mean", "sum", "count"]).sort_values("mean"))
log(feat.to_string())

# save the per case table for the appendix
per_case.to_csv(log_dir / "h2b_per_case_consistency.csv", index=False)

# Visualizations
sns.set_theme(style="whitegrid")

# Plot 1: share of consistent vs inconsistent cases
fig, ax = plt.subplots(figsize=(6, 5))
vals = [n_cases - n_inconsistent, n_inconsistent]
ax.bar(["consistent", "inconsistent"], vals, color=["#31a354", "#de2d26"])
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontweight="bold")
ax.set_title("H2b: Consistent vs inconsistent explanation codes (5 runs each)")
ax.set_ylabel("Number of cases")
fig.tight_layout()
fig.savefig(fig_dir / "h2b_consistency_counts.png", dpi=200)

# Plot 2: distribution of majority share
fig, ax = plt.subplots(figsize=(6, 5))
sns.histplot(per_case["majority_share"], bins=[0.2, 0.4, 0.6, 0.8, 1.01], ax=ax, color="#2c7fb8")
ax.set_title("H2b: Distribution of within-case majority share")
ax.set_xlabel("Majority share of the dominant code across 5 runs")
ax.set_ylabel("Number of cases")
fig.tight_layout()
fig.savefig(fig_dir / "h2b_majority_share_hist.png", dpi=200)

# Plot 3: agreement metrics per model with reference line at perfect agreement
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(model_df))
w = 0.27
ax.bar(x - w, model_df["kappa"], w, label="Fleiss kappa", color="#2c7fb8")
ax.bar(x, model_df["alpha"], w, label="Krippendorff alpha", color="#de2d26")
ax.bar(x + w, model_df["overlap"], w, label="Pairwise overlap", color="#756bb1")
ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="perfect agreement")
ax.set_xticks(x)
ax.set_xticklabels(model_df["model"], rotation=20, ha="right")
ax.set_ylabel("Agreement")
ax.set_title("H2b: Inter-run agreement of explanation codes per model")
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "h2b_agreement_per_model.png", dpi=200)

# Plot 4: consistency rate per perturbed feature
fig, ax = plt.subplots(figsize=(7, 5))
feat_plot = feat.reset_index()
ax.barh(feat_plot["perturbed_feature"], feat_plot["mean"], color="#2c7fb8")
ax.set_xlim(0, 1)
ax.set_title("H2b: Share of fully consistent cases per feature")
ax.set_xlabel("Consistency rate")
ax.set_ylabel("Perturbed feature")
fig.tight_layout()
fig.savefig(fig_dir / "h2b_consistency_per_feature.png", dpi=200)

# Plot 5: heatmap of code consistency by profile and model
piv_inc = per_case.pivot_table(index="profile_id", columns="model",
                               values="consistent", aggfunc="min")
fig, ax = plt.subplots(figsize=(8, 16))
sns.heatmap(piv_inc, cmap="RdYlGn", yticklabels=True,
            cbar_kws={"label": "1 = consistent, 0 = inconsistent"},
            linewidths=0.5, ax=ax)
ax.tick_params(axis="y", labelsize=7)
ax.set_title("H2b: Explanation code consistency by profile and model")
fig.tight_layout()
fig.savefig(fig_dir / "h2b_consistency_heatmap.png", dpi=200)