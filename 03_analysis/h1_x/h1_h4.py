"""
Exploratory H1 + H4 analysis

This script extends the original, working H1-H4 analysis and examines:
1. Whether score sensitivity differs between behavioral and firmographic
   features across the three language models.
2. Whether explanation alignment (final_code == 3) differs between behavioral
   and firmographic features across the three language models.
3. Whether explanation length differs between behavioral and firmographic
   features across the three language models.

The analyses are exploratory and separate from the primary H1-H4 tests.

Outputs
-------
1. Descriptive rates and figures for score sensitivity and alignment.
2. Descriptive explanation-length statistics and a figure.
3. GEE coefficient tables for all three outcomes.
4. Joint robust Wald tests of the Feature Type x Model interactions.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.proportion import proportion_confint

# 1. Paths and general settings
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 50)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "preprocessing"
SCORE_PATH = DATA_DIR / "llm_lead_intent_results_clean.csv"
EXPLANATION_PATH = DATA_DIR / "llm_lead_intent_results_explanations_coded.csv"

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
LOG_DIR = BASE_DIR / "logs"
for directory in (RESULTS_DIR, FIGURES_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "behavioral": "#2C7FB8",
    "firmographic": "#D95F0E",
}
BEHAVIORAL = ["website_dwell_time", "demo_requests", "email_response"]
FIRMOGRAPHIC = ["company_size", "industry", "region"]
FEATURES = BEHAVIORAL + FIRMOGRAPHIC
SCORE_MAP = {"Low Intent": 0, "High Intent": 1}


# 2. Logging
class Tee:
    """Write output to the terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


log_file = open(LOG_DIR / "h1_h4_model_interaction.log", "w", encoding="utf-8")
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)


# 3. Helper functions
def wilson(successes, total, alpha=0.05):
    """Return a Wilson confidence interval for a binomial proportion."""
    if total == 0:
        return np.nan, np.nan
    low, high = proportion_confint(successes, total, alpha=alpha, method="wilson")
    return float(low), float(high)


def rate_table(data, group_cols, outcome):
    """Calculate counts, rates, and descriptive Wilson intervals."""
    table = (
        data.groupby(group_cols, observed=True)[outcome]
        .agg(successes="sum", total="count")
        .reset_index()
    )
    table["rate"] = table["successes"] / table["total"]
    table[["ci_low", "ci_high"]] = pd.DataFrame(
        [wilson(int(s), int(n)) for s, n in zip(table["successes"], table["total"])],
        index=table.index,
    )
    table["ci_type"] = "Wilson descriptive; does not adjust for clustering"
    return table


def get_feature(perturbation):
    """Extract the manipulated feature from the perturbation label."""
    perturbation = str(perturbation)
    for feature in FEATURES:
        if perturbation.startswith(feature):
            return feature
    return "BASE"


def get_direction(perturbation):
    """Extract the perturbation direction from the perturbation label."""
    perturbation = str(perturbation)
    for direction in ("non-icp", "icp", "high", "low"):
        if perturbation.endswith(direction):
            return direction
    return "base"


def add_design_columns(data):
    """Derive profile, feature, direction, and feature-type variables."""
    data = data.copy()
    data["base_profile"] = data["profile_id"].astype(str).str.split("_").str[0]
    data["perturbation"] = (
        data["profile_id"].astype(str).str.split("_", n=1).str[1]
    )
    data["feature"] = data["perturbation"].apply(get_feature)
    data["direction"] = data["perturbation"].apply(get_direction)
    data["pole"] = data["direction"].map(
        {"high": "high", "icp": "high", "low": "low", "non-icp": "low"}
    )
    data["feature_type"] = np.select(
        [
            data["feature"].isin(BEHAVIORAL),
            data["feature"].isin(FIRMOGRAPHIC),
        ],
        ["behavioral", "firmographic"],
        default="base",
    )
    data["fe_behavioral"] = data["feature_type"].eq("behavioral").astype(int)
    data["cluster_pf"] = (
        data["base_profile"].astype(str) + "__" + data["feature"].astype(str)
    )
    return data


def save_gee_coefficients(result, analysis_name, filename, exponentiate=True):
    """Save GEE coefficients and robust confidence intervals."""
    confidence_intervals = result.conf_int()
    output = pd.DataFrame(
        {
            "analysis": analysis_name,
            "term": result.params.index,
            "estimate": result.params.to_numpy(),
            "robust_se": result.bse.to_numpy(),
            "ci_low": confidence_intervals.iloc[:, 0].to_numpy(),
            "ci_high": confidence_intervals.iloc[:, 1].to_numpy(),
            "p_value": result.pvalues.to_numpy(),
        }
    )
    if exponentiate:
        output["exp_estimate"] = np.exp(output["estimate"])
        output["exp_ci_low"] = np.exp(output["ci_low"])
        output["exp_ci_high"] = np.exp(output["ci_high"])
    output["n_observations"] = int(result.nobs)
    output["n_clusters"] = int(result.model.num_group)
    output.to_csv(RESULTS_DIR / filename, index=False)
    return output


def joint_interaction_test(result, interaction_pattern, analysis_name):
    """Run the same joint interaction test used in the original script."""
    parameter_names = list(result.params.index)
    interaction_terms = [
        name for name in parameter_names if interaction_pattern in name
    ]
    if not interaction_terms:
        raise RuntimeError(
            f"No Feature Type x Model interaction terms found for {analysis_name}."
        )

    restriction_matrix = np.zeros((len(interaction_terms), len(parameter_names)))
    for row_index, term in enumerate(interaction_terms):
        restriction_matrix[row_index, parameter_names.index(term)] = 1.0

    beta = result.params.to_numpy()
    covariance = result.cov_params().to_numpy()
    restricted_beta = restriction_matrix @ beta
    restricted_covariance = (
        restriction_matrix @ covariance @ restriction_matrix.T
    )

    # The interaction contains only two coefficients. Solve the 2x2 system
    # directly; use a very small diagonal regularization only if it is singular.
    try:
        solved = np.linalg.solve(restricted_covariance, restricted_beta)
        regularized = False
    except np.linalg.LinAlgError:
        scale = max(float(np.nanmax(np.abs(np.diag(restricted_covariance)))), 1.0)
        ridge = 1e-10 * scale
        solved = np.linalg.solve(
            restricted_covariance + ridge * np.eye(len(interaction_terms)),
            restricted_beta,
        )
        regularized = True

    wald_statistic = float(restricted_beta.T @ solved)
    degrees_of_freedom = len(interaction_terms)
    p_value = float(chi2.sf(wald_statistic, degrees_of_freedom))
    return {
        "analysis": analysis_name,
        "wald_chi_square": wald_statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "interaction_terms": " | ".join(interaction_terms),
        "n_observations": int(result.nobs),
        "n_clusters": int(result.model.num_group),
        "regularized_covariance": regularized,
        "note": "Joint robust Wald test of Feature Type x Model interaction coefficients",
    }


def plot_rate_figure(table, title, ylabel, filename):
    """Plot model-by-feature-type rates with descriptive Wilson intervals."""
    model_order = sorted(table["model"].unique())
    offsets = {"behavioral": -0.12, "firmographic": 0.12}
    markers = {"behavioral": "o", "firmographic": "s"}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for feature_type in ("behavioral", "firmographic"):
        rows = (
            table[table["feature_type"].eq(feature_type)]
            .set_index("model")
            .reindex(model_order)
        )
        x = np.arange(len(model_order)) + offsets[feature_type]
        rates = rows["rate"].to_numpy()
        ax.errorbar(
            x,
            rates,
            yerr=[rates - rows["ci_low"].to_numpy(), rows["ci_high"].to_numpy() - rates],
            fmt=markers[feature_type],
            linestyle="none",
            color=PALETTE[feature_type],
            ecolor=PALETTE[feature_type],
            markersize=8,
            elinewidth=1.5,
            capsize=5,
            label=feature_type.title(),
        )
    ax.set_xticks(np.arange(len(model_order)), model_order)
    ax.set(title=title, xlabel="Model", ylabel=ylabel, ylim=(0, 1))
    ax.legend(title="Feature type", loc="best")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# 4. Load and validate score data
if not SCORE_PATH.exists():
    raise FileNotFoundError(f"Input data not found: {SCORE_PATH}")
if not EXPLANATION_PATH.exists():
    raise FileNotFoundError(f"Input data not found: {EXPLANATION_PATH}")

score_df = pd.read_csv(SCORE_PATH)
required_score_columns = {"intent", "profile_id", "model", "run"}
missing = required_score_columns.difference(score_df.columns)
if missing:
    raise ValueError(f"Missing required score columns: {sorted(missing)}")

score_df["score"] = (
    score_df["intent"].astype(str).str.strip().map(SCORE_MAP)
)
if score_df["score"].isna().any():
    invalid = sorted(
        score_df.loc[score_df["score"].isna(), "intent"].astype(str).unique()
    )
    raise ValueError(f"Unrecognized intent values: {invalid}")
score_df = add_design_columns(score_df)


# 5. Score sensitivity by Feature Type x Model
perturbations = score_df[score_df["feature"] != "BASE"].copy()
pairs = (
    perturbations.pivot_table(
        index=["base_profile", "feature", "feature_type", "model", "run"],
        columns="pole",
        values="score",
        aggfunc="first",
    )
    .reset_index()
    .dropna(subset=["low", "high"])
)
pairs["delta"] = pairs["high"] - pairs["low"]
pairs["sensitivity"] = pairs["delta"].abs().eq(1).astype(int)
pairs["fe_behavioral"] = pairs["feature_type"].eq("behavioral").astype(int)
pairs["cluster_pf"] = (
    pairs["base_profile"].astype(str) + "__" + pairs["feature"].astype(str)
)

expected_pairs = (
    score_df["base_profile"].nunique()
    * len(FEATURES)
    * score_df["model"].nunique()
    * score_df["run"].nunique()
)
if len(pairs) != expected_pairs:
    print(f"WARNING: obtained {len(pairs)} matched pairs; expected {expected_pairs}.")

print("\nH1-H4 DATASET")
print(f"Matched Low-High pairs: {len(pairs)}")
print(f"Profile-feature clusters: {pairs['cluster_pf'].nunique()}")
print(f"Models: {pairs['model'].nunique()}")

sensitivity_rates = rate_table(
    pairs, ["model", "feature_type"], "sensitivity"
)
sensitivity_rates.to_csv(
    RESULTS_DIR / "h1_h4_score_sensitivity_descriptive.csv", index=False
)
print("\nDESCRIPTIVE SCORE-SENSITIVITY RATES")
print(sensitivity_rates.to_string(index=False))

plot_rate_figure(
    sensitivity_rates,
    "Score sensitivity by model and feature type",
    "Sensitivity rate",
    "h1_h4_fig1_score_sensitivity_by_model_and_type.png",
)

score_sensitivity_gee = GEE.from_formula(
    "sensitivity ~ fe_behavioral * C(model)",
    groups="cluster_pf",
    data=pairs,
    family=Binomial(),
    cov_struct=Exchangeable(),
).fit()
print("\nSCORE SENSITIVITY: FEATURE TYPE x MODEL GEE")
print(score_sensitivity_gee.summary())
save_gee_coefficients(
    score_sensitivity_gee,
    "Score sensitivity: Feature Type x Model",
    "h1_h4_score_sensitivity_coefficients.csv",
    exponentiate=True,
)
score_sensitivity_test = joint_interaction_test(
    score_sensitivity_gee,
    "fe_behavioral:C(model)",
    "Score sensitivity: Feature Type x Model interaction",
)


# 6. Load and prepare coded explanation data
explanation_df = pd.read_csv(EXPLANATION_PATH)
required_explanation_columns = {
    "profile_id", "run", "model", "reasoning", "final_code"
}
missing = required_explanation_columns.difference(explanation_df.columns)
if missing:
    raise ValueError(f"Missing required explanation columns: {sorted(missing)}")

explanation_df = add_design_columns(explanation_df)
explanation_df = explanation_df[explanation_df["feature"] != "BASE"].copy()
explanation_df["final_code"] = pd.to_numeric(
    explanation_df["final_code"], errors="coerce"
)
if explanation_df["final_code"].isna().any():
    raise ValueError("final_code contains missing or non-numeric values")
if not set(explanation_df["final_code"].astype(int).unique()).issubset({0, 1, 2, 3}):
    raise ValueError("final_code contains values outside 0, 1, 2, and 3")
explanation_df["aligned"] = explanation_df["final_code"].eq(3).astype(int)
explanation_df["word_count"] = (
    explanation_df["reasoning"]
    .fillna("")
    .astype(str)
    .str.findall(r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b")
    .str.len()
)

print("\nH1-H4 EXPLANATION DATASET")
print(f"Coded explanations: {len(explanation_df)}")
print(f"Exact perturbed profiles: {explanation_df['profile_id'].nunique()}")


# 7. Explanation alignment by Feature Type x Model
alignment_rates = rate_table(
    explanation_df, ["model", "feature_type"], "aligned"
)
alignment_rates.to_csv(
    RESULTS_DIR / "h1_h4_explanation_alignment_descriptive.csv", index=False
)
print("\nDESCRIPTIVE EXPLANATION-ALIGNMENT RATES")
print(alignment_rates.to_string(index=False))

plot_rate_figure(
    alignment_rates,
    "Explanation alignment by model and feature type",
    "Alignment rate (Code 3)",
    "h1_h4_fig2_explanation_alignment_by_model_and_type.png",
)

# profile_id clusters the repeated runs and same exact input across models.
explanation_alignment_gee = GEE.from_formula(
    "aligned ~ fe_behavioral * C(model)",
    groups="profile_id",
    data=explanation_df,
    family=Binomial(),
    cov_struct=Exchangeable(),
).fit()
print("\nEXPLANATION ALIGNMENT: FEATURE TYPE x MODEL GEE")
print(explanation_alignment_gee.summary())
save_gee_coefficients(
    explanation_alignment_gee,
    "Explanation alignment: Feature Type x Model",
    "h1_h4_explanation_alignment_coefficients.csv",
    exponentiate=True,
)
alignment_test = joint_interaction_test(
    explanation_alignment_gee,
    "fe_behavioral:C(model)",
    "Explanation alignment: Feature Type x Model interaction",
)


# 8. Explanation length by Feature Type x Model
length_descriptive = (
    explanation_df.groupby(["model", "feature_type"], observed=True)["word_count"]
    .agg(
        n="count",
        mean="mean",
        sd="std",
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        minimum="min",
        maximum="max",
    )
    .reset_index()
)
length_descriptive.to_csv(
    RESULTS_DIR / "h1_h4_explanation_length_descriptive.csv", index=False
)
print("\nDESCRIPTIVE EXPLANATION LENGTH")
print(length_descriptive.to_string(index=False))

# Log transformation reduces right skew. Gaussian GEE with robust standard errors
# avoids the convergence problems that can arise from a count-model GEE here.
explanation_df["log_word_count"] = np.log1p(explanation_df["word_count"])
explanation_length_gee = GEE.from_formula(
    "log_word_count ~ fe_behavioral * C(model)",
    groups="profile_id",
    data=explanation_df,
    family=Gaussian(),
    cov_struct=Exchangeable(),
).fit()
print("\nEXPLANATION LENGTH: FEATURE TYPE x MODEL GEE")
print(explanation_length_gee.summary())
save_gee_coefficients(
    explanation_length_gee,
    "Explanation length: Feature Type x Model",
    "h1_h4_explanation_length_coefficients.csv",
    exponentiate=True,
)
length_test = joint_interaction_test(
    explanation_length_gee,
    "fe_behavioral:C(model)",
    "Explanation length: Feature Type x Model interaction",
)

model_order = sorted(explanation_df["model"].unique())
fig, ax = plt.subplots(figsize=(9, 5.4))
positions = np.arange(len(model_order))
width = 0.34
for offset, feature_type in ((-width / 2, "behavioral"), (width / 2, "firmographic")):
    rows = (
        length_descriptive[length_descriptive["feature_type"].eq(feature_type)]
        .set_index("model")
        .reindex(model_order)
    )
    ax.bar(
        positions + offset,
        rows["mean"],
        width=width,
        yerr=rows["sd"],
        capsize=4,
        color=PALETTE[feature_type],
        label=feature_type.title(),
        alpha=0.9,
    )
ax.set_xticks(positions, model_order)
ax.set(
    title="Explanation length by model and feature type",
    xlabel="Model",
    ylabel="Mean word count",
)
ax.tick_params(axis="x", labelrotation=20)
ax.legend(title="Feature type")
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "h1_h4_fig3_explanation_length_by_model_and_type.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# 9. Consolidated interaction-test table
interaction_tests = pd.DataFrame(
    [score_sensitivity_test, alignment_test, length_test]
)
interaction_tests.to_csv(
    RESULTS_DIR / "h1_h4_interaction_tests.csv", index=False
)
print("\nJOINT FEATURE TYPE x MODEL INTERACTION TESTS")
print(interaction_tests.to_string(index=False))

sys.stdout = original_stdout
log_file.close()
