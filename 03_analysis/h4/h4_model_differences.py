"""
H4: Model differences in lead scores, perturbation sensitivity, output stability,
and explanations under identical inputs.

Analysis plan
-------------
H4a1: Overall High-Intent probability by model (all 975 rows, including baseline).
H4a2: Model differences in perturbation sensitivity (perturbed rows only), tested
       by the model × condition interaction.
H4b1: Lead-score stability by model using the Modal Agreement Rate (MAR), with
       baseline included. Exact input profiles are matched across models.
H4b2: Explanation-code stability by model using MAR (perturbed coded rows only).
H4c1: Explanation alignment by model, defined as final_code == 3.
H4c2: Exploratory distribution of Codes 0–3 by model. The codes are treated as
       nominal, not ordinal, because Code 2 is not an intermediate degree between
       Codes 1 and 3.

Notes
-----
- Model is always the focal predictor.
- Only five base profiles (A–E) exist. Therefore, base_profile is entered as a
  fixed blocking factor rather than estimated as a random effect.
- Binary row-level outcomes are analysed with Bayesian logistic mixed models
  (statsmodels BinomialBayesMixedGLM) with a random intercept for the identical
  input profile. Wald-style normal approximations based on posterior fixed-effect
  means/SDs are used for omnibus and pairwise summaries.
- MAR outcomes have few discrete levels and are perfectly matched by profile_id.
  Friedman tests and paired Wilcoxon tests are therefore used instead of forcing
  a Gaussian mixed model. Pairwise p-values are Holm-adjusted within each family.
- Baseline-only comparisons are descriptive because there are only five distinct
  baseline profiles. They are not used as separate confirmatory tests.
"""

from __future__ import annotations

import itertools
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2, friedmanchisquare, norm, wilcoxon
from statsmodels.stats.multitest import multipletests


# 0. Paths and logging
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FIG_DIR = SCRIPT_DIR / "figures"
LOG_DIR = SCRIPT_DIR / "logs"
RESULT_DIR = SCRIPT_DIR / "results"
for directory in (FIG_DIR, LOG_DIR, RESULT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SCORES_PATH = PROJECT_ROOT / "preprocessing" / "llm_lead_intent_results_clean.csv"
EXPL_PATH = PROJECT_ROOT / "preprocessing" / "llm_lead_intent_results_explanations_coded.csv"

logger = logging.getLogger("H4")
logger.setLevel(logging.INFO)
logger.handlers.clear()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler(LOG_DIR / "h4_analysis.log", mode="w", encoding="utf-8")
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

sns.set_theme(style="whitegrid", context="notebook")
ALPHA = 0.05


def log_df(title: str, frame: pd.DataFrame) -> None:
    logger.info("%s\n%s", title, frame.to_string(index=False))


# 1. Data preparation and validation
def parse_profile_id(profile_id: str) -> dict[str, str | bool]:
    """Parse base profile, feature, condition, and baseline status."""
    value = str(profile_id).strip()
    match = re.match(r"^([A-E])_(.+)$", value)
    if not match:
        raise ValueError(f"Unexpected profile_id format: {value!r}")

    base_profile, remainder = match.groups()
    if remainder.upper() == "BASE":
        return {
            "base_profile": base_profile,
            "feature": "baseline",
            "condition": "baseline",
            "is_baseline": True,
            "pair_id": f"{base_profile}_baseline",
        }

    # Parse suffixes from longest to shortest to avoid partial matches.
    suffix_map = {
        "non-icp": "unfavorable",
        "high": "favorable",
        "low": "unfavorable",
        "icp": "favorable",
    }
    for suffix in ("non-icp", "high", "low", "icp"):
        token = f"_{suffix}"
        if remainder.endswith(token):
            feature = remainder[: -len(token)]
            return {
                "base_profile": base_profile,
                "feature": feature,
                "condition": suffix_map[suffix],
                "is_baseline": False,
                "pair_id": f"{base_profile}_{feature}",
            }

    raise ValueError(f"Could not identify condition suffix in profile_id: {value!r}")


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SCORES_PATH.exists() or not EXPL_PATH.exists():
        raise FileNotFoundError(
            "Input files not found. Expected:\n"
            f"- {SCORES_PATH}\n- {EXPL_PATH}"
        )

    scores = pd.read_csv(SCORES_PATH)
    expl = pd.read_csv(EXPL_PATH)

    required_scores = {"profile_id", "run", "model", "intent"}
    required_expl = {"profile_id", "run", "model", "final_code"}
    if missing := required_scores.difference(scores.columns):
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    if missing := required_expl.difference(expl.columns):
        raise ValueError(f"Missing explanation columns: {sorted(missing)}")

    for frame in (scores, expl):
        frame["profile_id"] = frame["profile_id"].astype(str).str.strip()
        frame["model"] = frame["model"].astype(str).str.strip()
        parsed = frame["profile_id"].apply(parse_profile_id).apply(pd.Series)
        for column in parsed.columns:
            frame[column] = parsed[column]

    scores["score"] = scores["intent"].astype(str).str.strip().map(
        {"Low Intent": 0, "High Intent": 1}
    )
    if scores["score"].isna().any():
        bad = sorted(scores.loc[scores["score"].isna(), "intent"].astype(str).unique())
        raise ValueError(f"Unmapped intent values: {bad}")

    expl["final_code"] = pd.to_numeric(expl["final_code"], errors="coerce")
    if expl["final_code"].isna().any():
        raise ValueError("final_code contains missing or non-numeric values")
    expl["final_code"] = expl["final_code"].astype(int)
    invalid_codes = sorted(set(expl["final_code"]) - {0, 1, 2, 3})
    if invalid_codes:
        raise ValueError(f"Unexpected final_code values: {invalid_codes}")
    if expl["is_baseline"].any():
        logger.warning("Baseline rows found in explanation data; they will be excluded.")
        expl = expl.loc[~expl["is_baseline"]].copy()

    # Ensure one observation per profile × run × model.
    for name, frame in (("scores", scores), ("explanations", expl)):
        duplicates = frame.duplicated(["profile_id", "run", "model"]).sum()
        if duplicates:
            raise ValueError(f"{name} contains {duplicates} duplicate keys")

    models = sorted(scores["model"].unique())
    if len(models) != 3:
        logger.warning("Expected three models but found %d: %s", len(models), models)
    if set(expl["model"].unique()) != set(models):
        raise ValueError("Model sets differ between score and explanation files")

    logger.info("Loaded %d score rows and %d coded explanation rows", len(scores), len(expl))
    logger.info("Models: %s", models)
    logger.info("Base profiles: %s", sorted(scores["base_profile"].unique()))
    logger.info("Exact score profiles: %d", scores["profile_id"].nunique())
    return scores, expl


# 2. Logistic mixed-model helpers
def fit_logistic_glmm(
    data: pd.DataFrame,
    formula: str,
    cluster_column: str,
    analysis: str,
):
    """Fit a logistic random-intercept model using variational Bayes."""
    vc = {cluster_column: f"0 + C({cluster_column})"}
    model = sm.BinomialBayesMixedGLM.from_formula(formula, vc, data=data)
    result = model.fit_vb()
    logger.info("%s\n%s", analysis, result.summary())

    fixed = pd.DataFrame(
        {
            "analysis": analysis,
            "term": model.exog_names,
            "estimate_log_odds": result.fe_mean,
            "posterior_sd": result.fe_sd,
        }
    )
    fixed["odds_ratio"] = np.exp(fixed["estimate_log_odds"])
    fixed["ci_low"] = np.exp(fixed["estimate_log_odds"] - 1.96 * fixed["posterior_sd"])
    fixed["ci_high"] = np.exp(fixed["estimate_log_odds"] + 1.96 * fixed["posterior_sd"])
    fixed["z"] = fixed["estimate_log_odds"] / fixed["posterior_sd"]
    fixed["p_approx"] = 2 * norm.sf(np.abs(fixed["z"]))
    return model, result, fixed


def contrast_from_terms(
    model,
    result,
    weights: dict[str, float],
    analysis: str,
    contrast: str,
) -> dict:
    """Calculate a fixed-effect contrast using the VB diagonal covariance approximation."""
    names = list(model.exog_names)
    vector = np.array([weights.get(name, 0.0) for name in names], dtype=float)
    estimate = float(vector @ result.fe_mean)
    # fit_vb returns marginal posterior SDs; use the diagonal approximation explicitly.
    se = float(np.sqrt(np.sum((vector * result.fe_sd) ** 2)))
    z_value = estimate / se if se > 0 else np.nan
    p_value = 2 * norm.sf(abs(z_value)) if np.isfinite(z_value) else np.nan
    return {
        "analysis": analysis,
        "contrast": contrast,
        "estimate_log_odds": estimate,
        "se_approx": se,
        "odds_ratio": np.exp(estimate),
        "ci_low": np.exp(estimate - 1.96 * se),
        "ci_high": np.exp(estimate + 1.96 * se),
        "z_approx": z_value,
        "p_raw": p_value,
    }


def model_term(model_name: str, reference: str) -> str:
    return f"C(model, Treatment(reference='{reference}'))[T.{model_name}]"


def interaction_term(model_name: str, reference: str) -> str:
    base = model_term(model_name, reference)
    return f"{base}:C(condition, Treatment(reference='unfavorable'))[T.favorable]"


def model_pairwise_contrasts(model, result, models: list[str], analysis: str) -> pd.DataFrame:
    """Pairwise model contrasts for a model main effect under treatment coding."""
    reference = models[0]
    rows = []
    for left, right in itertools.combinations(models, 2):
        weights: dict[str, float] = {}
        if right != reference:
            weights[model_term(right, reference)] = 1.0
        if left != reference:
            weights[model_term(left, reference)] = -1.0
        rows.append(
            contrast_from_terms(
                model, result, weights, analysis, f"{right} - {left}"
            )
        )
    return holm_adjust(pd.DataFrame(rows))


def sensitivity_pairwise_contrasts(model, result, models: list[str]) -> pd.DataFrame:
    """Pairwise contrasts of favorable-vs-unfavorable effects between models."""
    reference = models[0]
    rows = []
    for left, right in itertools.combinations(models, 2):
        weights: dict[str, float] = {}
        if right != reference:
            weights[interaction_term(right, reference)] = 1.0
        if left != reference:
            weights[interaction_term(left, reference)] = -1.0
        rows.append(
            contrast_from_terms(
                model,
                result,
                weights,
                "H4a2_perturbation_sensitivity",
                f"Sensitivity({right}) - Sensitivity({left})",
            )
        )
    return holm_adjust(pd.DataFrame(rows))


def approximate_omnibus(model, result, terms: list[str], analysis: str) -> dict:
    """Approximate a joint Wald test using VB marginal posterior variances."""
    names = list(model.exog_names)
    indices = [names.index(term) for term in terms if term in names]
    if len(indices) != len(terms):
        missing = sorted(set(terms) - set(names))
        raise ValueError(f"Missing model terms for omnibus test: {missing}")
    z_squared = (result.fe_mean[indices] / result.fe_sd[indices]) ** 2
    statistic = float(np.sum(z_squared))
    df_test = len(indices)
    return {
        "analysis": analysis,
        "test": "Approximate joint Wald test (VB diagonal covariance)",
        "statistic": statistic,
        "df": df_test,
        "p_value": chi2.sf(statistic, df_test),
    }


def holm_adjust(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply Holm correction to one predefined pairwise-comparison family."""
    frame = frame.copy()
    valid = frame["p_raw"].notna()
    frame["p_holm"] = np.nan
    frame["reject_holm_0.05"] = False
    if valid.any():
        reject, corrected, _, _ = multipletests(
            frame.loc[valid, "p_raw"], alpha=ALPHA, method="holm"
        )
        frame.loc[valid, "p_holm"] = corrected
        frame.loc[valid, "reject_holm_0.05"] = reject
    return frame


# 3. Matched MAR helpers
def modal_agreement(
    data: pd.DataFrame, value_column: str, grouping: list[str]
) -> pd.DataFrame:
    """Calculate modal count and Modal Agreement Rate within repeated runs."""
    rows = []
    for keys, group in data.groupby(grouping, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        counts = group[value_column].value_counts()
        n_runs = int(counts.sum())
        modal_count = int(counts.max())
        row = dict(zip(grouping, keys))
        row.update(
            {
                "n_runs": n_runs,
                "modal_count": modal_count,
                "mar": modal_count / n_runs,
                "n_unique_outputs": int(counts.size),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def matched_mar_tests(mar: pd.DataFrame, analysis: str, models: list[str]) -> tuple[dict, pd.DataFrame]:
    """Run a Friedman omnibus test and paired Wilcoxon post-hoc comparisons."""
    wide = mar.pivot(index="profile_id", columns="model", values="mar").dropna()
    wide = wide.reindex(columns=models)
    if wide.empty:
        raise ValueError(f"No complete matched profiles for {analysis}")

    statistic, p_value = friedmanchisquare(*(wide[m].to_numpy() for m in models))
    omnibus = {
        "analysis": analysis,
        "test": "Friedman test on matched profile-level MAR",
        "statistic": statistic,
        "df": len(models) - 1,
        "p_value": p_value,
        "n_matched_profiles": len(wide),
    }

    rows = []
    for left, right in itertools.combinations(models, 2):
        differences = wide[right] - wide[left]
        if np.allclose(differences, 0):
            w_stat, p_raw = 0.0, 1.0
        else:
            w_stat, p_raw = wilcoxon(
                wide[right], wide[left], alternative="two-sided", zero_method="wilcox"
            )
        rows.append(
            {
                "analysis": analysis,
                "contrast": f"{right} - {left}",
                "mean_mar_difference": differences.mean(),
                "median_mar_difference": differences.median(),
                "wilcoxon_W": w_stat,
                "p_raw": p_raw,
                "n_matched_profiles": len(wide),
            }
        )
    return omnibus, holm_adjust(pd.DataFrame(rows))


# 4. Main analysis
def main() -> None:
    scores, expl = load_and_prepare()
    models = sorted(scores["model"].unique())
    reference = models[0]

    descriptive_tables = []
    fixed_effect_tables = []
    omnibus_rows = []
    pairwise_tables = []

    # H4a1: Overall High-Intent probability, including baseline profiles.
    desc_a1 = (
        scores.groupby("model", observed=True)["score"]
        .agg(high_intent_rate="mean", high_intent_n="sum", n="count")
        .reset_index()
    )
    desc_a1.insert(0, "analysis", "H4a1_overall_score")
    descriptive_tables.append(desc_a1)

    formula_a1 = (
        f"score ~ C(model, Treatment(reference='{reference}')) + C(base_profile) + "
        "C(is_baseline)"
    )
    mod_a1, res_a1, fixed_a1 = fit_logistic_glmm(
        scores, formula_a1, "profile_id", "H4a1_overall_score"
    )
    fixed_effect_tables.append(fixed_a1)
    terms_a1 = [model_term(model, reference) for model in models[1:]]
    omni_a1 = approximate_omnibus(mod_a1, res_a1, terms_a1, "H4a1_overall_score")
    omnibus_rows.append(omni_a1)
    if omni_a1["p_value"] < ALPHA:
        pairwise_tables.append(model_pairwise_contrasts(mod_a1, res_a1, models, "H4a1_overall_score"))
    else:
        logger.info("H4a1 omnibus not significant; pairwise comparisons skipped.")

    # H4a2: Perturbation sensitivity. Favorable/unfavorable is defined by suffix.
    perturbed = scores.loc[~scores["is_baseline"]].copy()
    desc_a2 = (
        perturbed.groupby(["model", "condition"], observed=True)["score"]
        .agg(high_intent_rate="mean", high_intent_n="sum", n="count")
        .reset_index()
    )
    desc_a2.insert(0, "analysis", "H4a2_perturbation_sensitivity")
    descriptive_tables.append(desc_a2)

    formula_a2 = (
        f"score ~ C(model, Treatment(reference='{reference}')) * "
        "C(condition, Treatment(reference='unfavorable')) + C(feature) + C(base_profile)"
    )
    mod_a2, res_a2, fixed_a2 = fit_logistic_glmm(
        perturbed, formula_a2, "pair_id", "H4a2_perturbation_sensitivity"
    )
    fixed_effect_tables.append(fixed_a2)
    interaction_terms = [interaction_term(model, reference) for model in models[1:]]
    omni_a2 = approximate_omnibus(
        mod_a2, res_a2, interaction_terms, "H4a2_perturbation_sensitivity"
    )
    omnibus_rows.append(omni_a2)
    if omni_a2["p_value"] < ALPHA:
        pairwise_tables.append(sensitivity_pairwise_contrasts(mod_a2, res_a2, models))
    else:
        logger.info("H4a2 interaction omnibus not significant; pairwise comparisons skipped.")

    # H4b1: Lead-score MAR, including baseline.
    score_mar = modal_agreement(
        scores,
        value_column="score",
        grouping=["profile_id", "base_profile", "feature", "is_baseline", "model"],
    )
    score_mar.to_csv(RESULT_DIR / "h4b1_score_mar_by_profile_model.csv", index=False)
    desc_b1 = (
        score_mar.groupby("model", observed=True)["mar"]
        .agg(mean_mar="mean", median_mar="median", sd_mar="std", n_profiles="count")
        .reset_index()
    )
    desc_b1.insert(0, "analysis", "H4b1_score_stability")
    descriptive_tables.append(desc_b1)
    omni_b1, pairs_b1 = matched_mar_tests(score_mar, "H4b1_score_stability", models)
    omnibus_rows.append(omni_b1)
    if omni_b1["p_value"] < ALPHA:
        pairwise_tables.append(pairs_b1)
    else:
        logger.info("H4b1 omnibus not significant; pairwise comparisons skipped.")

    # H4b2: Explanation-code MAR (no baselines exist in coded data).
    expl_mar = modal_agreement(
        expl,
        value_column="final_code",
        grouping=["profile_id", "base_profile", "feature", "model"],
    )
    expl_mar.to_csv(RESULT_DIR / "h4b2_explanation_mar_by_profile_model.csv", index=False)
    desc_b2 = (
        expl_mar.groupby("model", observed=True)["mar"]
        .agg(mean_mar="mean", median_mar="median", sd_mar="std", n_profiles="count")
        .reset_index()
    )
    desc_b2.insert(0, "analysis", "H4b2_explanation_stability")
    descriptive_tables.append(desc_b2)
    omni_b2, pairs_b2 = matched_mar_tests(expl_mar, "H4b2_explanation_stability", models)
    omnibus_rows.append(omni_b2)
    if omni_b2["p_value"] < ALPHA:
        pairwise_tables.append(pairs_b2)
    else:
        logger.info("H4b2 omnibus not significant; pairwise comparisons skipped.")

    # H4c1: Explanation alignment, exactly Code 3.
    expl["aligned"] = (expl["final_code"] == 3).astype(int)
    desc_c1 = (
        expl.groupby("model", observed=True)["aligned"]
        .agg(alignment_rate="mean", aligned_n="sum", n="count")
        .reset_index()
    )
    desc_c1.insert(0, "analysis", "H4c1_explanation_alignment")
    descriptive_tables.append(desc_c1)

    formula_c1 = (
        f"aligned ~ C(model, Treatment(reference='{reference}')) + C(feature) + "
        "C(condition) + C(base_profile)"
    )
    mod_c1, res_c1, fixed_c1 = fit_logistic_glmm(
        expl, formula_c1, "profile_id", "H4c1_explanation_alignment"
    )
    fixed_effect_tables.append(fixed_c1)
    terms_c1 = [model_term(model, reference) for model in models[1:]]
    omni_c1 = approximate_omnibus(
        mod_c1, res_c1, terms_c1, "H4c1_explanation_alignment"
    )
    omnibus_rows.append(omni_c1)
    if omni_c1["p_value"] < ALPHA:
        pairwise_tables.append(
            model_pairwise_contrasts(mod_c1, res_c1, models, "H4c1_explanation_alignment")
        )
    else:
        logger.info("H4c1 omnibus not significant; pairwise comparisons skipped.")

    # H4c2: Exploratory nominal Code 0–3 distribution.
    # Code 2 represents attribution on the rejected side and is not an ordinal
    # midpoint between Codes 1 and 3; an ordinal proportional-odds model would
    # therefore impose an invalid substantive ordering.
    code_counts = pd.crosstab(expl["model"], expl["final_code"]).reindex(
        index=models, columns=[0, 1, 2, 3], fill_value=0
    )
    code_rates = code_counts.div(code_counts.sum(axis=1), axis=0)
    code_counts.to_csv(RESULT_DIR / "h4c2_code_distribution_counts.csv")
    code_rates.to_csv(RESULT_DIR / "h4c2_code_distribution_rates.csv")

    code_long = (
        code_rates.rename_axis(index="model", columns="final_code")
        .reset_index()
        .melt(id_vars="model", var_name="final_code", value_name="rate")
    )
    code_long.insert(0, "analysis", "H4c2_code_distribution_exploratory")
    descriptive_tables.append(code_long)

    # Exploratory one-vs-rest mixed models provide code-specific model comparisons
    # without falsely treating Codes 0–3 as ordinal. These are descriptive/model-
    # based sensitivity analyses; no separate confirmatory claim is based on them.
    exploratory_code_rows = []
    for code in (0, 1, 2, 3):
        temp = expl.copy()
        temp["is_code"] = (temp["final_code"] == code).astype(int)
        formula_code = (
            f"is_code ~ C(model, Treatment(reference='{reference}')) + C(feature) + "
            "C(condition) + C(base_profile)"
        )
        mod_code, res_code, _ = fit_logistic_glmm(
            temp, formula_code, "profile_id", f"H4c2_exploratory_code_{code}"
        )
        terms_code = [model_term(model, reference) for model in models[1:]]
        omnibus_code = approximate_omnibus(
            mod_code, res_code, terms_code, f"H4c2_exploratory_code_{code}"
        )
        exploratory_code_rows.append(omnibus_code)
    exploratory_code_tests = pd.DataFrame(exploratory_code_rows)
    _, exploratory_code_tests["p_holm_across_codes"], _, _ = multipletests(
        exploratory_code_tests["p_value"], alpha=ALPHA, method="holm"
    )
    exploratory_code_tests.to_csv(
        RESULT_DIR / "h4c2_exploratory_code_specific_tests.csv", index=False
    )

    # Supplementary baseline analysis: descriptive only because n=5 base profiles.
    baseline = scores.loc[scores["is_baseline"]].copy()
    baseline_summary = (
        baseline.groupby(["base_profile", "model"], observed=True)["score"]
        .agg(high_intent_rate="mean", high_intent_n="sum", n_runs="count")
        .reset_index()
    )
    baseline_mar = score_mar.loc[score_mar["is_baseline"]].copy()
    baseline_summary = baseline_summary.merge(
        baseline_mar[["base_profile", "model", "mar", "modal_count"]],
        on=["base_profile", "model"],
        how="left",
    )
    baseline_summary.to_csv(
        RESULT_DIR / "supplementary_baseline_by_profile_model.csv", index=False
    )

    # Consolidate result tables.
    descriptive = pd.concat(descriptive_tables, ignore_index=True, sort=False)
    fixed_effects = pd.concat(fixed_effect_tables, ignore_index=True, sort=False)
    omnibus = pd.DataFrame(omnibus_rows)
    if pairwise_tables:
        pairwise = pd.concat(pairwise_tables, ignore_index=True, sort=False)
    else:
        pairwise = pd.DataFrame(
            columns=["analysis", "contrast", "p_raw", "p_holm", "reject_holm_0.05"]
        )

    descriptive.to_csv(RESULT_DIR / "h4_descriptive_statistics.csv", index=False)
    fixed_effects.to_csv(RESULT_DIR / "h4_glmm_fixed_effects.csv", index=False)
    omnibus.to_csv(RESULT_DIR / "h4_omnibus_tests.csv", index=False)
    pairwise.to_csv(RESULT_DIR / "h4_pairwise_comparisons_holm.csv", index=False)

    log_df("Omnibus tests", omnibus)
    if not pairwise.empty:
        log_df("Pairwise comparisons", pairwise)

    # 5. Figures
    # Overall High-Intent rates by model, split into baseline and perturbations.
    plot_scores = (
        scores.assign(input_type=np.where(scores["is_baseline"], "Baseline", "Perturbed"))
        .groupby(["model", "input_type"], observed=True)["score"]
        .mean()
        .reset_index(name="high_intent_rate")
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=plot_scores,
        x="model",
        y="high_intent_rate",
        hue="input_type",
        order=models,
        ax=ax,
    )
    ax.set(title="High-Intent rate by model and input type", xlabel="Model", ylabel="High-Intent rate", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "h4a1_high_intent_rate_by_model.png", dpi=300)
    plt.close(fig)

    # Perturbation sensitivity plot.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.pointplot(
        data=perturbed,
        x="condition",
        y="score",
        hue="model",
        hue_order=models,
        errorbar=("ci", 95),
        dodge=0.15,
        ax=ax,
    )
    ax.set(title="Perturbation sensitivity by model", xlabel="Perturbation condition", ylabel="High-Intent probability", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "h4a2_perturbation_sensitivity.png", dpi=300)
    plt.close(fig)

    # Score and explanation stability.
    for frame, title, filename in (
        (score_mar, "Lead-score stability by model", "h4b1_score_mar.png"),
        (expl_mar, "Explanation-code stability by model", "h4b2_explanation_mar.png"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.boxplot(data=frame, x="model", y="mar", order=models, ax=ax, color="#8fbcd4")
        sns.stripplot(data=frame, x="model", y="mar", order=models, ax=ax, color="black", alpha=0.35, size=3)
        ax.set(title=title, xlabel="Model", ylabel="Modal Agreement Rate", ylim=(0, 1.03))
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=300)
        plt.close(fig)

    # Alignment rate.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=expl, x="model", y="aligned", order=models, errorbar=("ci", 95), ax=ax, color="#4daf4a")
    ax.set(title="Explanation alignment rate by model (Code 3)", xlabel="Model", ylabel="Alignment rate", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "h4c1_alignment_rate.png", dpi=300)
    plt.close(fig)

    # Nominal explanation-code distribution.
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bottom = np.zeros(len(models))
    colors = ["#d73027", "#fc8d59", "#91bfdb", "#1a9850"]
    for code, color in zip((0, 1, 2, 3), colors):
        values = code_rates.loc[models, code].to_numpy()
        ax.bar(models, values, bottom=bottom, label=f"Code {code}", color=color)
        bottom += values
    ax.set(title="Distribution of explanation codes by model", xlabel="Model", ylabel="Proportion", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Explanation code", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "h4c2_code_distribution.png", dpi=300)
    plt.close(fig)

    # Baseline-only figure (five profiles; descriptive).
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=baseline_summary,
        x="base_profile",
        y="high_intent_rate",
        hue="model",
        hue_order=models,
        ax=ax,
    )
    ax.set(title="Baseline High-Intent rate by base profile and model (descriptive)", xlabel="Base profile", ylabel="High-Intent rate across five runs", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "supplementary_baseline_profiles.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
