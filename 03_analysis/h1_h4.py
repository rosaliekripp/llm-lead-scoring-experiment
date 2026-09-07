"""Supplementary H1-H4 analysis.

This script examines whether the behavioral-versus-firmographic sensitivity
difference varies between the three language models.

The analysis is separate from the primary H1 analysis.

Outputs
-------
1. Descriptive sensitivity rates by model and feature type.
2. A figure showing these rates with descriptive Wilson 95% intervals.
3. A logistic GEE with a FeatureType x Model interaction.
4. A joint Wald test of all FeatureType x Model interaction terms.
5. CSV result tables for descriptive and inferential results.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.proportion import proportion_confint


# 1. Paths and general settings

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 40)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR
    / "post_processing"
    / "llm_lead_intent_results_clean.csv"
)

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
LOG_DIR = BASE_DIR / "logs"

for directory in (RESULTS_DIR, FIGURES_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


PALETTE = {
    "behavioral": "#2C7FB8",
    "firmographic": "#D95F0E",
}

BEHAVIORAL = [
    "website_dwell_time",
    "demo_requests",
    "email_response",
]

FIRMOGRAPHIC = [
    "company_size",
    "industry",
    "region",
]

FEATURES = BEHAVIORAL + FIRMOGRAPHIC

SCORE_MAP = {
    "Low Intent": 0,
    "High Intent": 1,
}


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


log_file = open(
    LOG_DIR / "h1_h4_model_interaction.log",
    "w",
    encoding="utf-8",
)

original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)


# 3. Helper functions

def wilson(successes, total, alpha=0.05):
    """Return a Wilson confidence interval for a binomial proportion."""

    if total == 0:
        return np.nan, np.nan

    low, high = proportion_confint(
        successes,
        total,
        alpha=alpha,
        method="wilson",
    )

    return float(low), float(high)


def rate_table(data, group_cols, outcome):
    """Calculate counts, rates, and descriptive Wilson intervals."""

    table = (
        data
        .groupby(group_cols, observed=True)[outcome]
        .agg(
            successes="sum",
            total="count",
        )
        .reset_index()
    )

    table["rate"] = table["successes"] / table["total"]

    intervals = [
        wilson(int(successes), int(total))
        for successes, total
        in zip(table["successes"], table["total"])
    ]

    table[["ci_low", "ci_high"]] = pd.DataFrame(
        intervals,
        index=table.index,
    )

    table["ci_type"] = (
        "Wilson descriptive; does not adjust for clustering"
    )

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


# 4. Load and validate the data

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Input data not found: {DATA_PATH}. "
        "Place the cleaned CSV in post_processing/."
    )

df = pd.read_csv(DATA_PATH)

required_columns = {
    "intent",
    "profile_id",
    "model",
    "run",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


df["score"] = (
    df["intent"]
    .astype(str)
    .str.strip()
    .map(SCORE_MAP)
)

if df["score"].isna().any():
    invalid_values = sorted(
        df.loc[df["score"].isna(), "intent"]
        .astype(str)
        .unique()
    )

    raise ValueError(
        f"Unrecognized intent values: {invalid_values}"
    )


# 5. Create matched Low-High pairs

df["base_profile"] = (
    df["profile_id"]
    .astype(str)
    .str.split("_")
    .str[0]
)

df["perturbation"] = (
    df["profile_id"]
    .astype(str)
    .str.split("_", n=1)
    .str[1]
)

df["feature"] = df["perturbation"].apply(get_feature)
df["direction"] = df["perturbation"].apply(get_direction)

df["pole"] = df["direction"].map({
    "high": "high",
    "icp": "high",
    "low": "low",
    "non-icp": "low",
})

df["feature_type"] = np.select(
    [
        df["feature"].isin(BEHAVIORAL),
        df["feature"].isin(FIRMOGRAPHIC),
    ],
    [
        "behavioral",
        "firmographic",
    ],
    default="base",
)

perturbations = df[df["feature"] != "BASE"].copy()

pairs = (
    perturbations
    .pivot_table(
        index=[
            "base_profile",
            "feature",
            "feature_type",
            "model",
            "run",
        ],
        columns="pole",
        values="score",
        aggfunc="first",
    )
    .reset_index()
    .dropna(subset=["low", "high"])
)

pairs["delta"] = pairs["high"] - pairs["low"]

pairs["sensitivity"] = (
    pairs["delta"]
    .abs()
    .eq(1)
    .astype(int)
)

pairs["fe_behavioral"] = (
    pairs["feature_type"]
    .eq("behavioral")
    .astype(int)
)

# The cluster represents the same substantive profile-feature manipulation
# evaluated repeatedly across models and runs.
pairs["cluster_pf"] = (
    pairs["base_profile"].astype(str)
    + "__"
    + pairs["feature"].astype(str)
)


expected_pairs = (
    df["base_profile"].nunique()
    * len(FEATURES)
    * df["model"].nunique()
    * df["run"].nunique()
)

if len(pairs) != expected_pairs:
    print(
        f"WARNING: obtained {len(pairs)} matched pairs; "
        f"expected {expected_pairs}."
    )


print("\nH1-H4 ANALYSIS DATASET")
print(f"Matched Low-High pairs: {len(pairs)}")
print(
    "Profile-feature clusters: "
    f"{pairs['cluster_pf'].nunique()}"
)
print(f"Models: {pairs['model'].nunique()}")


# 6. Descriptive rates by model and feature type

model_type_rates = rate_table(
    pairs,
    ["model", "feature_type"],
    "sensitivity",
)

model_type_rates.to_csv(
    RESULTS_DIR / "h1_h4_descriptive_rates.csv",
    index=False,
)

print("\nDESCRIPTIVE SENSITIVITY RATES")
print(model_type_rates.to_string(index=False))


# 7. Descriptive H1-H4 figure

model_order = sorted(
    model_type_rates["model"].unique()
)

feature_type_order = [
    "behavioral",
    "firmographic",
]

feature_offsets = {
    "behavioral": -0.12,
    "firmographic": 0.12,
}

feature_markers = {
    "behavioral": "o",
    "firmographic": "s",
}


fig, ax = plt.subplots(figsize=(8.5, 5.2))

for feature_type in feature_type_order:

    feature_rows = (
        model_type_rates[
            model_type_rates["feature_type"].eq(feature_type)
        ]
        .set_index("model")
        .reindex(model_order)
    )

    x_positions = (
        np.arange(len(model_order))
        + feature_offsets[feature_type]
    )

    rates = feature_rows["rate"].to_numpy()
    ci_low = feature_rows["ci_low"].to_numpy()
    ci_high = feature_rows["ci_high"].to_numpy()

    ax.errorbar(
        x_positions,
        rates,
        yerr=[
            rates - ci_low,
            ci_high - rates,
        ],
        fmt=feature_markers[feature_type],
        linestyle="none",
        color=PALETTE[feature_type],
        ecolor=PALETTE[feature_type],
        markersize=8,
        elinewidth=1.5,
        capsize=5,
        label=feature_type.title(),
    )


ax.set_xticks(
    np.arange(len(model_order)),
    model_order,
)

ax.set(
    title="Sensitivity by model and feature type",
    xlabel="Model",
    ylabel="Sensitivity rate",
    ylim=(0, 1),
)

ax.legend(
    title="Feature type",
    loc="best",
)

ax.tick_params(
    axis="x",
    labelrotation=20,
)

fig.tight_layout()

fig.savefig(
    FIGURES_DIR / "h1_h4_sensitivity_by_model_and_type.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# 8. Logistic GEE with FeatureType x Model interaction

interaction_gee = GEE.from_formula(
    "sensitivity ~ fe_behavioral * C(model)",
    groups="cluster_pf",
    data=pairs,
    family=Binomial(),
    cov_struct=Exchangeable(),
).fit()

print("\nH1-H4 INTERACTION GEE")
print(interaction_gee.summary())


# 9. Save all GEE coefficients

confidence_intervals = interaction_gee.conf_int()

coefficient_results = pd.DataFrame({
    "term": interaction_gee.params.index,
    "log_odds": interaction_gee.params.to_numpy(),
    "robust_se": interaction_gee.bse.to_numpy(),
    "odds_ratio": np.exp(
        interaction_gee.params.to_numpy()
    ),
    "or_ci_low": np.exp(
        confidence_intervals.iloc[:, 0].to_numpy()
    ),
    "or_ci_high": np.exp(
        confidence_intervals.iloc[:, 1].to_numpy()
    ),
    "p_value": interaction_gee.pvalues.to_numpy(),
})

coefficient_results["n_observations"] = int(
    interaction_gee.nobs
)

coefficient_results["n_clusters"] = int(
    interaction_gee.model.num_group
)

coefficient_results.to_csv(
    RESULTS_DIR / "h1_h4_interaction_coefficients.csv",
    index=False,
)


# 10. Joint test of all FeatureType x Model interaction terms

parameter_names = list(
    interaction_gee.params.index
)

interaction_terms = [
    name
    for name in parameter_names
    if "fe_behavioral:C(model)" in name
]

if not interaction_terms:
    raise RuntimeError(
        "No FeatureType x Model interaction terms were found."
    )

restriction_matrix = np.zeros(
    (
        len(interaction_terms),
        len(parameter_names),
    )
)

for row_index, term in enumerate(interaction_terms):
    column_index = parameter_names.index(term)
    restriction_matrix[row_index, column_index] = 1.0


beta = interaction_gee.params.to_numpy()
covariance = interaction_gee.cov_params().to_numpy()

restricted_beta = restriction_matrix @ beta

restricted_covariance = (
    restriction_matrix
    @ covariance
    @ restriction_matrix.T
)

wald_statistic = float(
    restricted_beta.T
    @ np.linalg.pinv(restricted_covariance)
    @ restricted_beta
)

degrees_of_freedom = int(
    np.linalg.matrix_rank(restriction_matrix)
)

interaction_p_value = float(
    chi2.sf(
        wald_statistic,
        degrees_of_freedom,
    )
)


interaction_test = pd.DataFrame([{
    "analysis": "Joint FeatureType x Model interaction",
    "wald_chi_square": wald_statistic,
    "degrees_of_freedom": degrees_of_freedom,
    "p_value": interaction_p_value,
    "interaction_terms": " | ".join(interaction_terms),
    "n_observations": int(interaction_gee.nobs),
    "n_clusters": int(interaction_gee.model.num_group),
    "note": (
        "Joint robust Wald test of all FeatureType x Model "
        "interaction coefficients"
    ),
}])

interaction_test.to_csv(
    RESULTS_DIR / "h1_h4_interaction_test.csv",
    index=False,
)


print("\nJOINT FEATURE TYPE x MODEL INTERACTION TEST")
print(interaction_test.to_string(index=False))

sys.stdout = original_stdout
log_file.close()