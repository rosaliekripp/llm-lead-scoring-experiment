"""
H3: Explanation alignment.

Variant 1: perturbation cases whose majority score differs from the majority
           score of the corresponding baseline.
Variant 2: all perturbation cases.

The five API repetitions are NOT paired by run number. Scores are first reduced
to a majority score per experimental condition. Explanation alignment is then
measured per perturbation case as the share of its explanations assigned Code 3.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths and output folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "preprocessing"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

for directory in (FIGURES_DIR, LOGS_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CLEAN_FILE = DATA_DIR / "llm_lead_intent_results_clean.csv"
CODED_FILE = DATA_DIR / "llm_lead_intent_results_explanations_coded.csv"

# Settings
ALPHA = 0.05                 # 95% confidence intervals; not a test threshold
N_BOOTSTRAP = 10_000
RANDOM_SEED = 2026
EXPECTED_RUNS_PER_CASE = 5

# Logging
logger = logging.getLogger("h3")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(
    LOGS_DIR / "h3_explanation_alignment.log", mode="w", encoding="utf-8"
)
stream_handler = logging.StreamHandler()
for handler in (file_handler, stream_handler):
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def log(message: str = "") -> None:
    logger.info(message)


# Helpers
def require_columns(df: pd.DataFrame, columns: list[str], dataset: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {dataset}: {missing}")


def parse_condition_id(df: pd.DataFrame) -> pd.DataFrame:
    """Derive reference profile and perturbation condition from profile_id."""
    out = df.copy()
    parts = out["profile_id"].astype(str).str.split("_", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError(
            "profile_id must contain a reference profile and condition separated "
            "by an underscore, e.g. 'A_BASE'."
        )
    out["base_profile"] = parts[0]
    out["perturbation"] = parts[1]
    return out


def normalize_intent(series: pd.Series) -> pd.Series:
    """Convert supported binary score representations to 0/1."""
    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "low intent": 0,
        "high intent": 1,
        "low": 0,
        "high": 1,
        "0": 0,
        "1": 1,
    }
    labels = text.map(mapping)
    if labels.isna().any():
        unknown = sorted(text[labels.isna()].unique().tolist())
        raise ValueError(f"Unknown values in intent column: {unknown}")
    return labels.astype(int)


def majority_binary(series: pd.Series) -> int:
    """Majority of binary observations; an odd number of runs avoids ties."""
    values = pd.to_numeric(series, errors="raise").astype(int)
    if not values.isin([0, 1]).all():
        raise ValueError("Majority score can only be computed from 0/1 values.")
    if len(values) % 2 == 0 and values.mean() == 0.5:
        raise ValueError("Tie in majority score; define a tie rule before analysis.")
    return int(values.mean() > 0.5)


def bootstrap_mean_ci(
    values: pd.Series,
    alpha: float = ALPHA,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling whole perturbation cases."""
    array = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if len(array) == 0:
        return np.nan, np.nan
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(n_bootstrap, len(array)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [alpha / 2, 1 - alpha / 2]))


def summarize_variant(cases: pd.DataFrame, variant: str) -> dict[str, float | int | str]:
    n_cases = len(cases)
    n_explanations = int(cases["n_explanations"].sum()) if n_cases else 0
    n_code3 = int(cases["n_code3"].sum()) if n_cases else 0

    if n_cases:
        mean_share = float(cases["alignment_share"].mean())
        ci_low, ci_high = bootstrap_mean_ci(cases["alignment_share"])
        n_complete = int(cases["complete_alignment"].sum())
        complete_rate = n_complete / n_cases
    else:
        mean_share = ci_low = ci_high = np.nan
        n_complete = 0
        complete_rate = np.nan

    return {
        "variant": variant,
        "n_cases": n_cases,
        "n_explanations": n_explanations,
        "n_code3": n_code3,
        "mean_alignment_share": mean_share,
        "bootstrap_95ci_low": ci_low,
        "bootstrap_95ci_high": ci_high,
        "n_complete_alignment_5_of_5": n_complete,
        "complete_alignment_rate": complete_rate,
    }


# Load and validate data
if not CLEAN_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {CLEAN_FILE}")
if not CODED_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {CODED_FILE}")

scores = pd.read_csv(CLEAN_FILE)
coding = pd.read_csv(CODED_FILE)

require_columns(scores, ["profile_id", "model", "intent"], CLEAN_FILE.name)
require_columns(
    coding,
    ["profile_id", "model", "perturbed_feature", "final_code"],
    CODED_FILE.name,
)

scores = parse_condition_id(scores)
coding = parse_condition_id(coding)
scores["score"] = normalize_intent(scores["intent"])

coding["final_code"] = pd.to_numeric(coding["final_code"], errors="raise").astype(int)
if not coding["final_code"].isin([0, 1, 2, 3]).all():
    raise ValueError("final_code must only contain 0, 1, 2, or 3.")
if coding["perturbation"].str.upper().eq("BASE").any():
    raise ValueError("The coded explanation file unexpectedly contains baseline rows.")

coding["aligned"] = coding["final_code"].eq(3).astype(int)

# Determine score-changing cases by majority-score comparison with baseline
# No run-level pairing is used.
condition_scores = (
    scores.groupby(["profile_id", "model", "base_profile", "perturbation"], as_index=False)
    .agg(
        n_score_runs=("score", "size"),
        majority_score=("score", majority_binary),
        high_intent_share=("score", "mean"),
    )
)

baseline_scores = (
    condition_scores[condition_scores["perturbation"].str.upper().eq("BASE")]
    [["base_profile", "model", "majority_score", "n_score_runs"]]
    .rename(
        columns={
            "majority_score": "baseline_majority_score",
            "n_score_runs": "n_baseline_score_runs",
        }
    )
)

if baseline_scores.duplicated(["base_profile", "model"]).any():
    raise ValueError("More than one baseline condition found per profile and model.")

perturbation_scores = condition_scores[
    ~condition_scores["perturbation"].str.upper().eq("BASE")
].copy()
perturbation_scores = perturbation_scores.merge(
    baseline_scores,
    on=["base_profile", "model"],
    how="left",
    validate="many_to_one",
)
if perturbation_scores["baseline_majority_score"].isna().any():
    raise ValueError("At least one perturbation condition has no matching baseline.")

perturbation_scores["score_changed"] = (
    perturbation_scores["majority_score"]
    != perturbation_scores["baseline_majority_score"]
).astype(int)

# Aggregate explanation coding to the perturbation-case level
# One case = profile x model x perturbation condition.
case_coding = (
    coding.groupby(
        ["profile_id", "model", "base_profile", "perturbation", "perturbed_feature"],
        as_index=False,
    )
    .agg(
        n_explanations=("aligned", "size"),
        n_code3=("aligned", "sum"),
        alignment_share=("aligned", "mean"),
        n_code0=("final_code", lambda x: int((x == 0).sum())),
        n_code1=("final_code", lambda x: int((x == 1).sum())),
        n_code2=("final_code", lambda x: int((x == 2).sum())),
    )
)
case_coding["complete_alignment"] = (
    case_coding["n_code3"] == case_coding["n_explanations"]
).astype(int)

cases = case_coding.merge(
    perturbation_scores[
        [
            "profile_id",
            "model",
            "majority_score",
            "baseline_majority_score",
            "score_changed",
            "n_score_runs",
            "n_baseline_score_runs",
        ]
    ],
    on=["profile_id", "model"],
    how="left",
    validate="one_to_one",
)

if cases["score_changed"].isna().any():
    raise ValueError("At least one coded perturbation case has no matching score case.")

# Warnings are logged rather than silently changing the analysis.
if not cases["n_explanations"].eq(EXPECTED_RUNS_PER_CASE).all():
    log("WARNING: Not every perturbation case has exactly five coded explanations.")
if not cases["n_score_runs"].eq(EXPECTED_RUNS_PER_CASE).all():
    log("WARNING: Not every perturbation condition has exactly five score runs.")
if not cases["n_baseline_score_runs"].eq(EXPECTED_RUNS_PER_CASE).all():
    log("WARNING: Not every baseline condition has exactly five score runs.")

# H3 variants – descriptive analysis only
variant_1 = cases[cases["score_changed"].eq(1)].copy()
variant_2 = cases.copy()

summary = pd.DataFrame(
    [
        summarize_variant(
            variant_1,
            "V1: majority score differs from corresponding baseline",
        ),
        summarize_variant(
            variant_2,
            "V2: all perturbation cases",
        ),
    ]
)

# Complete Code 0–3 distribution, without feature/model comparisons.
code_distribution = pd.DataFrame(
    [
        {
            "variant": "V1",
            "code": code,
            "count": int((coding.merge(
                variant_1[["profile_id", "model"]],
                on=["profile_id", "model"],
                how="inner",
            )["final_code"] == code).sum()),
        }
        for code in range(4)
    ]
    + [
        {
            "variant": "V2",
            "code": code,
            "count": int((coding["final_code"] == code).sum()),
        }
        for code in range(4)
    ]
)
code_distribution["share"] = code_distribution.groupby("variant")["count"].transform(
    lambda x: x / x.sum()
)

# Save result tables
cases.to_csv(RESULTS_DIR / "h3_case_level_data.csv", index=False)
summary.to_csv(RESULTS_DIR / "h3_variant_summary.csv", index=False)
code_distribution.to_csv(RESULTS_DIR / "h3_code_distribution.csv", index=False)

log("H3 Explanation Alignment")
log("========================")
log("Definition of a score-changing case:")
log("  Majority score of perturbation != majority score of corresponding baseline")
log("  Run numbers are not paired.")
log("Alignment measure:")
log("  Share of the five coded explanations receiving Code 3.")
log("Complete alignment:")
log("  All available explanations in the case receive Code 3 (normally 5/5).")
log("No binomial test or consistency-threshold test is performed.")
log("")
log(summary.to_string(index=False))
log("\nCode distribution:")
log(code_distribution.to_string(index=False))


# Figures
plt.style.use("seaborn-v0_8-whitegrid")

# Figure 1: number of score-stable and score-changing perturbation cases
counts = cases["score_changed"].value_counts().reindex([0, 1], fill_value=0)
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["Score stable vs. baseline", "Score changed vs. baseline\n(Variant 1)"],
    counts.values,
    color=["#9ecae1", "#08519c"],
)
ax.bar_label(bars, padding=3, fontweight="bold")
ax.set_ylabel("Number of perturbation cases")
ax.set_title("Cases that did or did not change the score")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "h3_fig1_case_base.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 2: mean case-level alignment share with case-bootstrap 95% CIs
plot_data = summary.copy()
plot_data = plot_data.iloc[:2].reset_index(drop=True)
labels = [
    f"V1: score-changing\n(n={int(plot_data.loc[0, 'n_cases'])})",
    f"V2: all cases\n(n={int(plot_data.loc[1, 'n_cases'])})",
]
y = plot_data["mean_alignment_share"].to_numpy(dtype=float)
ci_low = plot_data["bootstrap_95ci_low"].to_numpy(dtype=float)
ci_high = plot_data["bootstrap_95ci_high"].to_numpy(dtype=float)
lower_error = np.maximum(0, y - ci_low)
upper_error = np.maximum(0, ci_high - y)
yerr = np.vstack([lower_error, upper_error])
fig, ax = plt.subplots(figsize=(7.5, 5.5))
bars = ax.bar(
    labels,
    y,
    yerr=yerr,
    capsize=6,
    width=0.72,
    color=["#08519c", "#2c7fb8"],
    edgecolor="white",
    linewidth=0.8,
    error_kw={
        "ecolor": "#222222",
        "elinewidth": 1.5,
        "capthick": 1.5,
    },
)
ax.axhline(
    y=1.0,
    color="#222222",
    linestyle=":",
    linewidth=1.8,
    label="Perfect alignment (1.00)",
    zorder=1,
)
for bar, value, upper_ci in zip(bars, y, ci_high):
    label_y = min(upper_ci + 0.025, 1.045)
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        label_y,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
ax.set_ylim(0, 1.08)
ax.set_ylabel("Mean share of explanations with Code 3")
ax.set_title("Explanation alignment")
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda value, _: f"{value:.0%}")
)
ax.legend(
    loc="upper right",
    frameon=True,
    fontsize=9,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.text(
    0.5,
    0.015,
    "Error bars show 95% case-bootstrap confidence intervals.",
    ha="center",
    va="bottom",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(
    FIGURES_DIR / "h3_fig2_alignment_variants.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)