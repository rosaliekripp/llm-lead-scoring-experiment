"""
H2 – Stability analysis
=======================
Analyses score stability and explanation-attribution stability across five
repeated runs of identical model inputs.

Primary analyses only:
- Score: intent (Low Intent / High Intent)
- Explanation attribution: final_code (0–3, treated as nominal categories)

Explicitly not included:
- Fleiss' kappa
- Mention stability (0 vs. 1–3)
- Alignment stability (3 vs. 0–2)
- Inferential model comparisons (reserved for H4)

Analyzed data:
- Score file: 975 rows = 195 identical-input/model units × 5 runs
- Explanation file: 900 rows = 180 perturbation/model units × 5 runs

A stability unit is one unique input condition evaluated by one model.
The script searches for a condition-ID column that, together with `model`,
uniquely identifies groups of exactly five runs. Set CONDITION_COLUMN manually
below if automatic detection is not suitable for the dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from krippendorff import alpha as krippendorff_alpha


# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "post_processing"
TABLE_DIR = BASE_DIR / "tables"
FIGURE_DIR = BASE_DIR / "figures"
LOG_DIR = BASE_DIR / "logs"

SCORE_FILE = DATA_DIR / "llm_lead_intent_results_clean.csv"
EXPLANATION_FILE = DATA_DIR / "llm_lead_intent_results_explanations_coded.csv"

CONDITION_COLUMN = "profile_id"

MODEL_COLUMN = "model"
RUN_COLUMN = "run"
SCORE_COLUMN = "intent"
EXPLANATION_CODE_COLUMN = "final_code"

EXPECTED_RUNS = 5
EXPECTED_SCORE_ROWS = 975
EXPECTED_EXPLANATION_ROWS = 900
EXPECTED_SCORE_UNITS = 195
EXPECTED_EXPLANATION_UNITS = 180

N_BOOTSTRAP = 5_000
RANDOM_SEED = 42
ALPHA_LEVEL = 0.05

# Candidate names, in preferred order. `profile_id` is accepted only if it
# actually identifies complete conditions such as A_demo_requests_high.
CONDITION_COLUMN_CANDIDATES = [
    "case_id",
    "condition_id",
    "input_id",
    "profile_case_id",
    "profile_id",
    "case",
]

SCORE_MAP = {
    "low intent": 0,
    "high intent": 1,
}


# Logging and directories

for directory in (BASE_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("h2_stability")
logger.setLevel(logging.INFO)
logger.handlers.clear()
formatter = logging.Formatter("%(message)s")

file_handler = logging.FileHandler(
    LOG_DIR / "h2_stability_analysis.log", mode="w", encoding="utf-8"
)
console_handler = logging.StreamHandler()
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def log(message: str = "") -> None:
    logger.info(message)


# Data validation and preparation


def require_columns(df: pd.DataFrame, columns: list[str], dataset_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: missing required column(s): {', '.join(missing)}"
        )


def choose_condition_column(df: pd.DataFrame, dataset_name: str) -> str:
    """Find a condition column that forms groups of exactly five runs with model."""
    if CONDITION_COLUMN is not None:
        if CONDITION_COLUMN not in df.columns:
            raise ValueError(
                f"{dataset_name}: configured CONDITION_COLUMN "
                f"'{CONDITION_COLUMN}' does not exist."
            )
        candidates = [CONDITION_COLUMN]
    else:
        candidates = [
            column for column in CONDITION_COLUMN_CANDIDATES if column in df.columns
        ]

    if not candidates:
        raise ValueError(
            f"{dataset_name}: no condition-ID column found. Set CONDITION_COLUMN "
            "at the top of the script to the column containing identifiers such "
            "as 'A_demo_requests_high'."
        )

    for candidate in candidates:
        keys = [candidate, MODEL_COLUMN]
        duplicate_run = df.duplicated(keys + [RUN_COLUMN], keep=False)
        group_sizes = df.groupby(keys, dropna=False).size()
        if not duplicate_run.any() and group_sizes.eq(EXPECTED_RUNS).all():
            return candidate

    diagnostics = []
    for candidate in candidates:
        keys = [candidate, MODEL_COLUMN]
        duplicated = int(df.duplicated(keys + [RUN_COLUMN], keep=False).sum())
        sizes = df.groupby(keys, dropna=False).size()
        invalid = int((sizes != EXPECTED_RUNS).sum())
        diagnostics.append(
            f"{candidate}: duplicated condition/model/run rows={duplicated}, "
            f"groups not containing {EXPECTED_RUNS} runs={invalid}"
        )

    raise ValueError(
        f"{dataset_name}: none of the candidate columns uniquely identifies an "
        f"input condition with {EXPECTED_RUNS} runs per model.\n"
        + "\n".join(diagnostics)
        + "\nSet CONDITION_COLUMN manually to the correct column."
    )


def validate_repeated_structure(
    df: pd.DataFrame,
    condition_column: str,
    expected_rows: int,
    expected_units: int,
    dataset_name: str,
) -> list[str]:
    unit_columns = [condition_column, MODEL_COLUMN]

    if len(df) != expected_rows:
        raise ValueError(
            f"{dataset_name}: expected {expected_rows} rows, found {len(df)}."
        )

    if df[unit_columns + [RUN_COLUMN]].isna().any().any():
        raise ValueError(
            f"{dataset_name}: missing values found in condition, model, or run columns."
        )

    duplicated = df.duplicated(unit_columns + [RUN_COLUMN], keep=False)
    if duplicated.any():
        example = df.loc[duplicated, unit_columns + [RUN_COLUMN]].head(10)
        raise ValueError(
            f"{dataset_name}: duplicate observations per stability unit and run. "
            f"Examples:\n{example.to_string(index=False)}"
        )

    group_sizes = df.groupby(unit_columns, dropna=False).size()
    invalid_sizes = group_sizes[group_sizes != EXPECTED_RUNS]
    if not invalid_sizes.empty:
        raise ValueError(
            f"{dataset_name}: every unit must contain exactly {EXPECTED_RUNS} runs. "
            f"Invalid groups:\n{invalid_sizes.head(10).to_string()}"
        )

    n_units = len(group_sizes)
    if n_units != expected_units:
        raise ValueError(
            f"{dataset_name}: expected {expected_units} stability units, found {n_units}."
        )

    run_counts = df.groupby(unit_columns, dropna=False)[RUN_COLUMN].nunique()
    if not run_counts.eq(EXPECTED_RUNS).all():
        raise ValueError(f"{dataset_name}: run identifiers are not unique within units.")

    return unit_columns


def load_score_data() -> tuple[pd.DataFrame, list[str], str]:
    df = pd.read_csv(SCORE_FILE)
    require_columns(df, [MODEL_COLUMN, RUN_COLUMN, SCORE_COLUMN], "Score data")

    condition_column = choose_condition_column(df, "Score data")
    unit_columns = validate_repeated_structure(
        df,
        condition_column,
        EXPECTED_SCORE_ROWS,
        EXPECTED_SCORE_UNITS,
        "Score data",
    )

    normalized = df[SCORE_COLUMN].astype("string").str.strip().str.lower()
    df["score"] = normalized.map(SCORE_MAP)
    invalid = df.loc[df["score"].isna(), SCORE_COLUMN].drop_duplicates().tolist()
    if invalid:
        raise ValueError(
            "Score data: `intent` contains unsupported values. Expected "
            f"'Low Intent' or 'High Intent'; found: {invalid}"
        )
    df["score"] = df["score"].astype(int)
    return df, unit_columns, condition_column


def load_explanation_data() -> tuple[pd.DataFrame, list[str], str]:
    df = pd.read_csv(EXPLANATION_FILE)
    require_columns(
        df,
        [MODEL_COLUMN, RUN_COLUMN, EXPLANATION_CODE_COLUMN],
        "Explanation data",
    )

    condition_column = choose_condition_column(df, "Explanation data")
    unit_columns = validate_repeated_structure(
        df,
        condition_column,
        EXPECTED_EXPLANATION_ROWS,
        EXPECTED_EXPLANATION_UNITS,
        "Explanation data",
    )

    numeric_codes = pd.to_numeric(df[EXPLANATION_CODE_COLUMN], errors="coerce")
    if numeric_codes.isna().any():
        bad = df.loc[numeric_codes.isna(), EXPLANATION_CODE_COLUMN].drop_duplicates()
        raise ValueError(
            "Explanation data: `final_code` contains missing/non-numeric values: "
            f"{bad.tolist()}"
        )
    if not numeric_codes.isin([0, 1, 2, 3]).all():
        bad = sorted(numeric_codes.loc[~numeric_codes.isin([0, 1, 2, 3])].unique())
        raise ValueError(
            f"Explanation data: `final_code` must contain only 0, 1, 2, 3; found {bad}."
        )
    df["explanation_code"] = numeric_codes.astype(int)
    return df, unit_columns, condition_column


# Stability metrics


def make_reliability_matrix(
    df: pd.DataFrame, unit_columns: list[str], value_column: str
) -> pd.DataFrame:
    """Rows = stability units, columns = repeated runs; pivot fails on duplicates."""
    matrix = df.pivot(
        index=unit_columns,
        columns=RUN_COLUMN,
        values=value_column,
    ).sort_index(axis=1)

    if matrix.shape[1] != EXPECTED_RUNS or matrix.isna().any().any():
        raise ValueError(
            f"Invalid reliability matrix for {value_column}: expected exactly "
            f"{EXPECTED_RUNS} complete run columns."
        )
    return matrix


def nominal_alpha(matrix: pd.DataFrame | np.ndarray) -> float:
    values = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    # krippendorff expects observers/runs in rows and units in columns.
    result = krippendorff_alpha(
        reliability_data=values.T,
        level_of_measurement="nominal",
    )
    return float(result)


def bootstrap_nominal_alpha(
    matrix: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, int]:
    """Bootstrap complete stability units; all five runs remain together."""
    rng = np.random.default_rng(seed)
    values = matrix.to_numpy()
    n_units = values.shape[0]
    estimates: list[float] = []

    for _ in range(n_bootstrap):
        sampled_indices = rng.integers(0, n_units, size=n_units)
        estimate = nominal_alpha(values[sampled_indices, :])
        if np.isfinite(estimate):
            estimates.append(estimate)

    if not estimates:
        return np.nan, np.nan, 0

    lower = float(np.quantile(estimates, ALPHA_LEVEL / 2))
    upper = float(np.quantile(estimates, 1 - ALPHA_LEVEL / 2))
    return lower, upper, len(estimates)


def unit_statistics(matrix: pd.DataFrame, outcome: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for unit_id, row in matrix.iterrows():
        values = row.to_numpy()
        counts = pd.Series(values).value_counts().sort_index()
        max_count = int(counts.max())
        modal_categories = counts[counts == max_count].index.tolist()
        modal_share = max_count / len(values)

        result: dict[str, Any] = {
            "outcome": outcome,
            "n_runs": len(values),
            "n_unique_categories": int(counts.size),
            "fully_stable": int(counts.size == 1),
            "any_instability": int(counts.size > 1),
            "modal_share": modal_share,
            "instability": 1.0 - modal_share,
            "n_modal_categories": len(modal_categories),
            "mode_tie": int(len(modal_categories) > 1),
            "modal_categories": "|".join(map(str, modal_categories)),
            "run_pattern": "|".join(map(str, values.tolist())),
        }

        if isinstance(unit_id, tuple):
            for name, value in zip(matrix.index.names, unit_id):
                result[name] = value
        else:
            result[matrix.index.name or "unit"] = unit_id

        if outcome == "score":
            n_high = int(np.sum(values == 1))
            n_low = int(np.sum(values == 0))
            result["n_high"] = n_high
            result["n_low"] = n_low
            result["agreement_split"] = f"{max(n_high, n_low)}:{min(n_high, n_low)}"
        else:
            for code in [0, 1, 2, 3]:
                result[f"n_code_{code}"] = int(np.sum(values == code))

        rows.append(result)

    return pd.DataFrame(rows)


def summarise_stability(
    matrix: pd.DataFrame,
    per_unit: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    alpha_value = nominal_alpha(matrix)
    ci_low, ci_high, valid_bootstraps = bootstrap_nominal_alpha(matrix)
    n_units = len(per_unit)
    n_stable = int(per_unit["fully_stable"].sum())
    n_unstable = int(per_unit["any_instability"].sum())

    return pd.DataFrame(
        [
            {
                "outcome": outcome,
                "n_units": n_units,
                "n_runs_per_unit": EXPECTED_RUNS,
                "n_fully_stable": n_stable,
                "fully_stable_rate": n_stable / n_units,
                "n_any_instability": n_unstable,
                "any_instability_rate": n_unstable / n_units,
                "mean_modal_share": per_unit["modal_share"].mean(),
                "mean_instability": per_unit["instability"].mean(),
                "n_mode_ties": int(per_unit["mode_tie"].sum()),
                "mode_tie_rate": per_unit["mode_tie"].mean(),
                "krippendorff_alpha_nominal": alpha_value,
                "alpha_ci_95_low": ci_low,
                "alpha_ci_95_high": ci_high,
                "valid_bootstrap_samples": valid_bootstraps,
            }
        ]
    )


# Output tables, logging, and figures


def log_summary(summary: pd.DataFrame, title: str) -> None:
    row = summary.iloc[0]
    log(title)
    log("=" * len(title))
    log(f"Stability units: {int(row['n_units'])}")
    log(
        f"Fully stable: {int(row['n_fully_stable'])}/{int(row['n_units'])} "
        f"({row['fully_stable_rate']:.3f})"
    )
    log(
        f"Any instability: {int(row['n_any_instability'])}/{int(row['n_units'])} "
        f"({row['any_instability_rate']:.3f})"
    )
    log(f"Mean modal share: {row['mean_modal_share']:.3f}")
    log(f"Mean instability (1 - modal share): {row['mean_instability']:.3f}")
    log(
        "Krippendorff's alpha (nominal): "
        f"{row['krippendorff_alpha_nominal']:.3f}, 95% bootstrap CI "
        f"[{row['alpha_ci_95_low']:.3f}, {row['alpha_ci_95_high']:.3f}]"
    )
    if row["outcome"] == "explanation":
        log(
            f"Units with tied modes: {int(row['n_mode_ties'])}/{int(row['n_units'])} "
            f"({row['mode_tie_rate']:.3f})"
        )
    log()


def save_distribution_tables(
    score_units: pd.DataFrame, explanation_units: pd.DataFrame
) -> None:
    score_split = (
        score_units["agreement_split"]
        .value_counts()
        .reindex(["5:0", "4:1", "3:2"], fill_value=0)
        .rename_axis("agreement_split")
        .reset_index(name="n_units")
    )
    score_split["proportion"] = score_split["n_units"] / len(score_units)
    score_split.to_csv(TABLE_DIR / "h2_score_agreement_distribution.csv", index=False)

    explanation_modal = (
        explanation_units["modal_share"]
        .value_counts()
        .sort_index(ascending=False)
        .rename_axis("modal_share")
        .reset_index(name="n_units")
    )
    explanation_modal["proportion"] = (
        explanation_modal["n_units"] / len(explanation_units)
    )
    explanation_modal.to_csv(
        TABLE_DIR / "h2_explanation_modal_share_distribution.csv", index=False
    )


def create_figures(
    score_units: pd.DataFrame, explanation_units: pd.DataFrame
) -> None:
    sns.set_theme(style="whitegrid")

    # Score: exact 5:0, 4:1, and 3:2 split across repeated runs.
    split_counts = (
        score_units["agreement_split"]
        .value_counts()
        .reindex(["5:0", "4:1", "3:2"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ca25f", "#fec44f", "#de2d26"]
    bars = ax.bar(split_counts.index, split_counts.values, color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_title("Score stability across five identical runs")
    ax.set_xlabel("Majority-to-minority score split")
    ax.set_ylabel("Number of stability units")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "h2_fig1_score_agreement_distribution.png", dpi=300)
    plt.close(fig)

    # Explanation: complete stability versus any code instability.
    explanation_counts = pd.Series(
        {
            "Fully stable": int(explanation_units["fully_stable"].sum()),
            "Any instability": int(explanation_units["any_instability"].sum()),
        }
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        explanation_counts.index,
        explanation_counts.values,
        color=["#2ca25f", "#de2d26"],
    )
    ax.bar_label(bars, padding=3)
    ax.set_title("Stability of explanation-attribution codes")
    ax.set_ylabel("Number of stability units")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "h2_fig2_explanation_complete_stability.png", dpi=300)
    plt.close(fig)

    # Explanation: distribution of the modal-code share.
    modal_counts = explanation_units["modal_share"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [f"{value:.1f}" for value in modal_counts.index],
        modal_counts.values,
        color="#3182bd",
    )
    ax.bar_label(bars, padding=3)
    ax.set_title("Modal-code share across five identical runs")
    ax.set_xlabel("Modal-code share")
    ax.set_ylabel("Number of stability units")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "h2_fig3_explanation_modal_share.png", dpi=300)
    plt.close(fig)


def main() -> None:
    log("H2 stability analysis")
    log("=====================")
    log()

    score_df, score_unit_columns, score_condition_column = load_score_data()
    explanation_df, explanation_unit_columns, explanation_condition_column = (
        load_explanation_data()
    )

    log(f"Score condition-ID column: {score_condition_column}")
    log(f"Explanation condition-ID column: {explanation_condition_column}")
    log()

    score_matrix = make_reliability_matrix(
        score_df, score_unit_columns, "score"
    )
    explanation_matrix = make_reliability_matrix(
        explanation_df, explanation_unit_columns, "explanation_code"
    )

    score_units = unit_statistics(score_matrix, outcome="score")
    explanation_units = unit_statistics(
        explanation_matrix, outcome="explanation"
    )

    score_summary = summarise_stability(
        score_matrix, score_units, outcome="score"
    )
    explanation_summary = summarise_stability(
        explanation_matrix, explanation_units, outcome="explanation"
    )
    combined_summary = pd.concat(
        [score_summary, explanation_summary], ignore_index=True
    )

    log_summary(score_summary, "H2a – Score stability")
    log_summary(explanation_summary, "H2b – Explanation-attribution stability")

    combined_summary.to_csv(TABLE_DIR / "h2_stability_summary.csv", index=False)
    score_units.to_csv(TABLE_DIR / "h2_score_per_unit.csv", index=False)
    explanation_units.to_csv(
        TABLE_DIR / "h2_explanation_per_unit.csv", index=False
    )

    save_distribution_tables(score_units, explanation_units)
    create_figures(score_units, explanation_units)


if __name__ == "__main__":
    main()
