"""
H4: The generated lead scores, their stability, and the associated explanations
    differ significantly between language models given identical input data.

Design note (per supervisor):
- The 5 runs per (profile x model) are repeated measurements and are NOT independent.
- 'model' is a FIXED effect inside mixed-effects models with a RANDOM INTERCEPT
  per profile. Because the intent outcome is binary, a logistic GLMM is used.
- Multiplicity: Holm as primary correction; Benjamini-Hochberg (FDR) as a
  less-conservative fallback (supervisor's suggestion).

Explanation alignment:
- Primary operationalization: aligned = final_code >= 3 (perturbed feature was a
  supporting / load-bearing factor for the intent).
- Sensitivity check: aligned = final_code >= 1 (feature merely mentioned).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal, chi2_contingency

# ------------------------------------------------------------------ #
# 0. Logging setup  (console + file, like the figures in h1)
# ------------------------------------------------------------------ #
base_dir = Path(__file__).parent
log_dir = base_dir / "logs"
fig_dir = base_dir / "figures"
log_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

log_path = log_dir / "h4_model_comparison.log"

logger = logging.getLogger("H4")
logger.setLevel(logging.INFO)
logger.handlers.clear()  # avoid duplicate handlers on re-run

_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(_fmt)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(_fmt)
logger.addHandler(console_handler)

def log_df(title, frame):
    """Log a dataframe as a monospaced block."""
    logger.info("%s\n%s", title, frame.to_string())

logger.info("=" * 70)
logger.info("H4 model comparison — analysis started")
logger.info("Log file: %s", log_path)
logger.info("=" * 70)

# ------------------------------------------------------------------ #
# 1. Load data
# ------------------------------------------------------------------ #
df = pd.read_csv(base_dir / "post_processing" / "llm_lead_intent_results_clean.csv")
expl = pd.read_csv(base_dir / "post_processing" / "llm_lead_intent_results_explanations_coded.csv")
logger.info("Loaded scores: %d rows | explanation coding: %d rows",
            len(df), len(expl))

# ------------------------------------------------------------------ #
# 2. Preparation
# ------------------------------------------------------------------ #
score_map = {"Low Intent": 0, "High Intent": 1}
df["score"] = df["intent"].str.strip().map(score_map)
df["base_profile"] = df["profile_id"].str.split("_").str[0]
df["model"] = df["model"].str.strip()

n_unmapped = df["score"].isna().sum()
if n_unmapped:
    logger.warning("%d intent values could not be mapped to 0/1 and are dropped",
                   n_unmapped)

logger.info("Models found: %s", sorted(df["model"].unique()))
logger.info("Runs per model: %s", df.groupby("model")["run"].count().to_dict())

pvalue_registry = []  # (label, p)

def register(label, p):
    pvalue_registry.append((label, float(p)))
    logger.info("REGISTERED p-value [%s] raw p = %.4g", label, p)


# ================================================================== #
# H4a: Lead scores differ between models  (logistic GLMM)
# ================================================================== #
logger.info("-" * 70)
logger.info("H4a: Lead scores differ between models (logistic GLMM)")
logger.info("-" * 70)

df_glmm = df.dropna(subset=["score"]).copy()
vc = {"profile": "0 + C(profile_id)"}  # random intercept per profile

try:
    binom_glmm = sm.BinomialBayesMixedGLM.from_formula(
        "score ~ C(model)", vc_formulas=vc, data=df_glmm
    )
    res_a = binom_glmm.fit_vb()
    logger.info("H4a logistic GLMM summary:\n%s", res_a.summary())
except Exception as e:
    logger.exception("H4a GLMM failed, continuing with robust omnibus only: %s", e)

# robust omnibus on independent per-profile mean scores
per_unit = (df_glmm.groupby(["model", "profile_id"])["score"].mean().reset_index())
groups = [g["score"].values for _, g in per_unit.groupby("model")]
H_a, p_a = kruskal(*groups)
logger.info("H4a omnibus (Kruskal-Wallis on per-profile mean scores): "
            "H=%.3f, p=%.4g", H_a, p_a)
register("H4a_scores_between_models", p_a)


# ================================================================== #
# H4b: Score stability differs between models
# instability = within-cell variance of the binary score across the 5 runs
# ================================================================== #
logger.info("-" * 70)
logger.info("H4b: Score stability differs between models")
logger.info("-" * 70)

stab = (df_glmm.groupby(["model", "profile_id", "base_profile"])["score"]
        .agg(["mean", "var", "count"]).reset_index())
stab["instability"] = stab["var"].fillna(0.0)
stab["flipped"] = stab["mean"].between(0.0, 1.0, inclusive="neither").astype(int)

log_df("Instability (within-cell variance) per model:",
       stab.groupby("model")["instability"].agg(["mean", "std", "count"]))

try:
    mlm_b = smf.mixedlm("instability ~ C(model)", stab,
                        groups=stab["base_profile"]).fit()
    logger.info("H4b mixed model summary:\n%s", mlm_b.summary())
except Exception as e:
    logger.exception("H4b mixed model failed: %s", e)

groups_b = [g["instability"].values for _, g in stab.groupby("model")]
H_b, p_b = kruskal(*groups_b)
logger.info("H4b omnibus (Kruskal-Wallis on instability): H=%.3f, p=%.4g", H_b, p_b)
register("H4b_stability_between_models", p_b)


# ================================================================== #
# H4c: Explanation alignment differs between models
# primary: aligned = final_code >= 3 ; sensitivity: >= 1
# ================================================================== #
logger.info("-" * 70)
logger.info("H4c: Explanation alignment differs between models")
logger.info("-" * 70)

expl["model"] = expl["model"].str.strip()
expl["final_code"] = pd.to_numeric(expl["final_code"], errors="coerce")
n_bad = expl["final_code"].isna().sum()
if n_bad:
    logger.warning("%d rows have non-numeric final_code and are dropped", n_bad)
expl = expl.dropna(subset=["final_code"]).copy()

def run_alignment_test(threshold, label):
    """Run the alignment analysis for a given 'aligned' threshold."""
    logger.info(">> Alignment operationalization: aligned = final_code >= %d",
                threshold)
    e = expl.copy()
    e["aligned"] = (e["final_code"] >= threshold).astype(int)

    log_df(f"Alignment rate per model (>= {threshold}):",
           e.groupby("model")["aligned"].agg(["mean", "sum", "count"]))

    # primary: logistic GLMM, random intercept per profile
    try:
        glmm_c = sm.BinomialBayesMixedGLM.from_formula(
            "aligned ~ C(model)", vc_formulas={"profile": "0 + C(profile_id)"},
            data=e,
        )
        res_c = glmm_c.fit_vb()
        logger.info("H4c logistic GLMM summary (>= %d):\n%s",
                    threshold, res_c.summary())
    except Exception as ex:
        logger.exception("H4c GLMM failed (>= %d): %s", threshold, ex)

    # supporting omnibus: chi-square (marginal)
    ct = pd.crosstab(e["model"], e["aligned"])
    chi2, p_chi, dof, _ = chi2_contingency(ct)
    log_df(f"Contingency table model x aligned (>= {threshold}):", ct)
    logger.info("H4c chi-square (>= %d): chi2=%.3f, dof=%d, p=%.4g",
                threshold, chi2, dof, p_chi)

    # robust clustered omnibus: KW on per-profile alignment rate
    per_unit_c = (e.groupby(["model", "profile_id"])["aligned"].mean().reset_index())
    groups_c = [g["aligned"].values for _, g in per_unit_c.groupby("model")]
    H_c, p_kw = kruskal(*groups_c)
    logger.info("H4c omnibus (Kruskal-Wallis on per-profile alignment rate, "
                ">= %d): H=%.3f, p=%.4g", threshold, H_c, p_kw)
    register(label, p_kw)

# primary (>=3) and sensitivity (>=1)
run_alignment_test(3, "H4c_alignment_between_models_primary_ge3")
run_alignment_test(1, "H4c_alignment_between_models_sensitivity_ge1")

# ================================================================== #
# 2d. Figures  (saved under base_dir / "figures")
# ================================================================== #
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint

logger.info("-" * 70)
logger.info("Generating figures -> %s", fig_dir)
logger.info("-" * 70)

sns.set_theme(style="whitegrid")

def _wilson(series):
    """Return (rate, err_low, err_high) with Wilson 95% CI for a 0/1 series."""
    k, n = int(series.sum()), int(series.count())
    rate = k / n if n else 0.0
    low, high = proportion_confint(k, n, alpha=0.05, method="wilson")
    return rate, rate - low, high - rate

# order models consistently across all plots
model_order = sorted(df_glmm["model"].unique())

# ---------- Figure 1 (H4a): High-Intent rate per model with Wilson CI ----------
rows = []
for m, g in df_glmm.groupby("model"):
    rate, el, eh = _wilson(g["score"])
    rows.append({"model": m, "rate": rate, "err_low": el, "err_high": eh})
agg_a = pd.DataFrame(rows).set_index("model").reindex(model_order).reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(agg_a["model"], agg_a["rate"],
       yerr=[agg_a["err_low"], agg_a["err_high"]], capsize=6, color="#2c7fb8")
for i, r in agg_a.iterrows():
    ax.text(i, r["rate"] + r["err_high"] + 0.02, f"{r['rate']:.1%}",
            ha="center", va="bottom", fontweight="bold")
ax.set_title("H4a: High-Intent rate per model (Wilson 95% CI)")
ax.set_xlabel("Model"); ax.set_ylabel("Share of 'High Intent' runs")
ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(fig_dir / "h4a_score_rate_per_model.png", dpi=200)
plt.close(fig)

# ---------- Figure 2 (H4b): Score instability per model ----------
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=stab, x="model", y="instability", order=model_order,
            color="#de2d26", ax=ax)
sns.stripplot(data=stab, x="model", y="instability", order=model_order,
              color="black", alpha=0.35, size=3, ax=ax)
ax.set_title("H4b: Score instability per model (within-cell variance across runs)")
ax.set_xlabel("Model"); ax.set_ylabel("Instability (variance of 0/1 score)")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(fig_dir / "h4b_instability_per_model.png", dpi=200)
plt.close(fig)

# ---------- Figure 2b (H4b): Flip rate per model (share of non-unanimous cells) ----------
rows = []
for m, g in stab.groupby("model"):
    rate, el, eh = _wilson(g["flipped"])
    rows.append({"model": m, "rate": rate, "err_low": el, "err_high": eh})
agg_flip = pd.DataFrame(rows).set_index("model").reindex(model_order).reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(agg_flip["model"], agg_flip["rate"],
       yerr=[agg_flip["err_low"], agg_flip["err_high"]], capsize=6, color="#756bb1")
for i, r in agg_flip.iterrows():
    ax.text(i, r["rate"] + r["err_high"] + 0.02, f"{r['rate']:.1%}",
            ha="center", va="bottom", fontweight="bold")
ax.set_title("H4b: Flip rate per model (share of profiles with non-unanimous runs)")
ax.set_xlabel("Model"); ax.set_ylabel("Flip rate")
ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(fig_dir / "h4b_flip_rate_per_model.png", dpi=200)
plt.close(fig)

# ---------- Figure 3 (H4c): Alignment rate per model, primary (>=3) vs sensitivity (>=1) ----------
rows = []
for thr in (3, 1):
    e = expl.copy()
    e["aligned"] = (e["final_code"] >= thr).astype(int)
    for m, g in e.groupby("model"):
        rate, el, eh = _wilson(g["aligned"])
        rows.append({"model": m, "threshold": f">= {thr}",
                     "rate": rate, "err_low": el, "err_high": eh})
agg_c = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=agg_c, x="model", y="rate", hue="threshold",
            order=model_order, palette={">= 3": "#31a354", ">= 1": "#a1d99b"},
            ax=ax)
ax.set_title("H4c: Explanation alignment rate per model (primary >=3 vs. sensitivity >=1)")
ax.set_xlabel("Model"); ax.set_ylabel("Alignment rate")
ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=30)
ax.legend(title="Aligned if final_code")
fig.tight_layout()
fig.savefig(fig_dir / "h4c_alignment_rate_per_model.png", dpi=200)
plt.close(fig)

logger.info("Figures saved: %s",
            [p.name for p in sorted(fig_dir.glob('h4*.png'))])


# ================================================================== #
# 3. Multiplicity correction:  Holm (primary) + Benjamini-Hochberg (fallback)
# ================================================================== #
logger.info("=" * 70)
logger.info("Multiple-testing correction across H4 sub-tests")
logger.info("=" * 70)

labels = [lbl for lbl, _ in pvalue_registry]
pvals = [p for _, p in pvalue_registry]

holm_rej, holm_p, _, _ = multipletests(pvals, alpha=0.05, method="holm")
fdr_rej, fdr_p, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

summary = pd.DataFrame({
    "test": labels,
    "p_raw": pvals,
    "p_holm": holm_p,
    "sig_holm(.05)": holm_rej,
    "p_fdr_bh": fdr_p,
    "sig_fdr(.05)": fdr_rej,
})
log_df("H4 corrected results:", summary)

out_csv = base_dir / "logs" / "h4_model_comparison_results.csv"
summary.to_csv(out_csv, index=False)
logger.info("Saved results table -> %s", out_csv)
logger.info("Saved full log       -> %s", log_path)
logger.info("H4 analysis finished.")