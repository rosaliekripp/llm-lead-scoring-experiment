"""
Exploratory H1 + H2 analysis
----------------------------
Examines whether score stability and explanation-attribution stability differ
between behavioral and firmographic feature perturbations.

A condition-model unit is unstable if its five identical runs contain:
- more than one `intent` value (score instability), or
- more than one `final_code` value (explanation instability).

For each outcome, behavioral and firmographic perturbations are compared using
Fisher's exact test. Individual features are reported descriptively only.
The tests are exploratory and unadjusted; they do not model repeated structures.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact


# Configuration

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "preprocessing"
SCORE_FILE = DATA_DIR / "llm_lead_intent_results_clean.csv"
EXPLANATION_FILE = DATA_DIR / "llm_lead_intent_results_explanations_coded.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
LOG_DIR = BASE_DIR / "logs"
for directory in (RESULTS_DIR, FIGURES_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Column containing identifiers such as A_demo_requests_high.
CONDITION_COLUMN = "profile_id"
MODEL_COLUMN = "model"
RUN_COLUMN = "run"
INTENT_COLUMN = "intent"
EXPLANATION_COLUMN = "final_code"
EXPECTED_RUNS = 5

BEHAVIORAL_FEATURES = {"website_dwell_time", "demo_requests", "email_response"}
FIRMOGRAPHIC_FEATURES = {"company_size", "industry", "region"}


# Logging

logger = logging.getLogger("h1_h2_feature_stability")
logger.setLevel(logging.INFO)
logger.handlers.clear()
for handler in (
    logging.FileHandler(
        LOG_DIR / "h1_h2_feature_stability.log", mode="w", encoding="utf-8"
    ),
    logging.StreamHandler(),
):
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def log(message=""):
    logger.info(message)


# Preparation


def extract_feature(condition):
    """Extract the perturbed feature from IDs such as A_demo_requests_high."""
    value = str(condition).strip()
    upper = value.upper()
    if upper.endswith("_BASE") or upper == "BASE":
        return "BASE"

    parts = value.split("_", 1)
    perturbation = parts[1] if len(parts) == 2 else parts[0]
    for suffix in ("_high", "_low", "_icp", "_non-icp", "_non_icp"):
        if perturbation.lower().endswith(suffix):
            return perturbation[: -len(suffix)].lower()
    raise ValueError(f"Cannot extract feature from condition ID: {value}")


def validate_and_prepare(df, value_column, outcome):
    required = {CONDITION_COLUMN, MODEL_COLUMN, RUN_COLUMN, value_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{outcome}: missing required columns: {sorted(missing)}")

    df = df.copy()
    df["feature"] = df[CONDITION_COLUMN].apply(extract_feature)
    df = df[df["feature"] != "BASE"].copy()

    known_features = BEHAVIORAL_FEATURES | FIRMOGRAPHIC_FEATURES
    unknown = sorted(set(df["feature"]) - known_features)
    if unknown:
        raise ValueError(
            f"{outcome}: unassigned features found: {unknown}. Check mappings."
        )

    df["feature_type"] = df["feature"].map(
        lambda x: "Behavioral" if x in BEHAVIORAL_FEATURES else "Firmographic"
    )

    unit_columns = [CONDITION_COLUMN, MODEL_COLUMN]
    if df.duplicated(unit_columns + [RUN_COLUMN]).any():
        raise ValueError(
            f"{outcome}: duplicate observation for condition, model, and run."
        )

    group_sizes = df.groupby(unit_columns).size()
    if not group_sizes.eq(EXPECTED_RUNS).all():
        invalid = group_sizes[group_sizes != EXPECTED_RUNS]
        raise ValueError(
            f"{outcome}: every condition-model unit must contain exactly five "
            f"runs. Invalid groups:\n{invalid.head(10).to_string()}"
        )

    if df[value_column].isna().any():
        raise ValueError(f"{outcome}: missing values found in `{value_column}`.")

    return df


def load_score_data():
    df = pd.read_csv(SCORE_FILE)
    required = {INTENT_COLUMN}
    if not required.issubset(df.columns):
        raise ValueError(f"Score: missing column `{INTENT_COLUMN}`.")

    labels = df[INTENT_COLUMN].astype(str).str.strip().str.lower().map(
        {"low intent": 0, "high intent": 1}
    )
    if labels.isna().any():
        invalid = df.loc[labels.isna(), INTENT_COLUMN].drop_duplicates().tolist()
        raise ValueError(f"Score: unsupported intent values: {invalid}")
    df["analysis_value"] = labels.astype(int)
    return validate_and_prepare(df, "analysis_value", "Score")


def load_explanation_data():
    df = pd.read_csv(EXPLANATION_FILE)
    if EXPLANATION_COLUMN not in df.columns:
        raise ValueError(f"Explanation: missing column `{EXPLANATION_COLUMN}`.")

    codes = pd.to_numeric(df[EXPLANATION_COLUMN], errors="coerce")
    if codes.isna().any():
        invalid = df.loc[codes.isna(), EXPLANATION_COLUMN].drop_duplicates().tolist()
        raise ValueError(f"Explanation: invalid final_code values: {invalid}")
    if not codes.isin([0, 1, 2, 3]).all():
        invalid = sorted(codes.loc[~codes.isin([0, 1, 2, 3])].unique())
        raise ValueError(f"Explanation: final_code must be 0–3; found {invalid}.")
    df["analysis_value"] = codes.astype(int)
    return validate_and_prepare(df, "analysis_value", "Explanation")


# Analysis


def create_stability_units(df, outcome):
    """Create one row per identical-input/model unit."""
    unit_columns = [CONDITION_COLUMN, MODEL_COLUMN]
    units = (
        df.groupby(unit_columns, as_index=False)
        .agg(
            feature=("feature", "first"),
            feature_type=("feature_type", "first"),
            n_unique_values=("analysis_value", "nunique"),
            n_runs=("analysis_value", "size"),
            run_pattern=(
                "analysis_value",
                lambda values: "|".join(map(str, values.tolist())),
            ),
        )
    )
    units["outcome"] = outcome
    units["unstable"] = (units["n_unique_values"] > 1).astype(int)
    return units


def analyse_outcome(units, outcome):
    contingency = pd.crosstab(
        units["feature_type"], units["unstable"]
    ).reindex(index=["Behavioral", "Firmographic"], columns=[0, 1], fill_value=0)
    contingency.columns = ["stable", "unstable"]

    odds_ratio, p_value = fisher_exact(
        contingency[["stable", "unstable"]].to_numpy()
    )

    group_summary = (
        units.groupby("feature_type")["unstable"]
        .agg(n_units="count", n_unstable="sum", instability_rate="mean")
        .reindex(["Behavioral", "Firmographic"])
        .reset_index()
    )
    group_summary.insert(0, "outcome", outcome)
    group_summary["fisher_odds_ratio"] = odds_ratio
    group_summary["fisher_p_value"] = p_value

    feature_summary = (
        units.groupby(["feature_type", "feature"])["unstable"]
        .agg(n_units="count", n_unstable="sum", instability_rate="mean")
        .reset_index()
        .sort_values(["feature_type", "instability_rate"], ascending=[True, False])
    )
    feature_summary.insert(0, "outcome", outcome)

    log(f"{outcome} stability")
    log("-" * (len(outcome) + 10))
    log("Contingency table:")
    log(contingency.to_string())
    log("\nInstability by feature type:")
    log(group_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log(f"\nFisher's exact test: OR = {odds_ratio:.3f}, p = {p_value:.4f}")
    log("\nDescriptive instability by individual feature:")
    log(feature_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    log()

    return group_summary, feature_summary, contingency


# Visualization


def create_figure(group_summary):
    sns.set_theme(style="whitegrid")
    plot_data = group_summary.copy()
    plot_data["Outcome"] = plot_data["outcome"].map(
        {"Score": "Lead score", "Explanation": "Explanation attribution"}
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.barplot(
        data=plot_data,
        x="feature_type",
        y="instability_rate",
        hue="Outcome",
        palette=["#2c7fb8", "#d95f0e"],
        ax=ax,
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{bar.get_height():.1%}" for bar in container],
            padding=3,
        )

    maximum = plot_data["instability_rate"].max()
    ax.set_ylim(0, max(0.10, maximum * 1.30))
    ax.set_xlabel("Perturbed feature type")
    ax.set_ylabel("Share of unstable units")
    ax.set_title("Output instability by perturbed feature type")
    ax.legend(title="Outcome")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "h1_h2_instability_by_feature_type.png", dpi=300
    )
    plt.close(fig)


# Main


def main():
    score_df = load_score_data()
    explanation_df = load_explanation_data()

    score_units = create_stability_units(score_df, "Score")
    explanation_units = create_stability_units(explanation_df, "Explanation")

    log("Exploratory H1 + H2 analysis: instability by feature type")
    log("==========================================================")
    log(f"Score perturbation units: {len(score_units)}")
    log(f"Explanation perturbation units: {len(explanation_units)}")
    log()

    score_group, score_feature, score_table = analyse_outcome(
        score_units, "Score"
    )
    explanation_group, explanation_feature, explanation_table = analyse_outcome(
        explanation_units, "Explanation"
    )

    all_units = pd.concat([score_units, explanation_units], ignore_index=True)
    all_groups = pd.concat([score_group, explanation_group], ignore_index=True)
    all_features = pd.concat([score_feature, explanation_feature], ignore_index=True)

    all_units.to_csv(RESULTS_DIR / "h1_h2_stability_units.csv", index=False)
    all_groups.to_csv(RESULTS_DIR /  "h1_h2_feature_type_summary.csv", index=False)
    all_features.to_csv(
        RESULTS_DIR /  "h1_h2_individual_feature_summary.csv", index=False
    )

    score_table.to_csv(RESULTS_DIR /  "h1_h2_score_contingency.csv", index=False)
    explanation_table.to_csv(
        RESULTS_DIR /  "h1_h2_explanation_contingency.csv", index=False
    )

    create_figure(all_groups)
    log("Note: Both Fisher tests are exploratory and unadjusted.")


if __name__ == "__main__":
    main()