"""
Exploratory H1 + H3 analysis: explanation alignment by feature.

Outcome
-------
Explanation alignment remains defined as Code 3. For each perturbation case,
the outcome is the share of its five explanations assigned Code 3.

Scope
-----
The primary complementary analysis uses all perturbation cases. It reports:
1. case-level alignment by each of the six manipulated features;
2. alignment for behavioral versus firmographic features; and
3. a paired, two-sided exact sign-flip permutation test for the difference
   between feature types.

Pairing
-------
Behavioral and firmographic means are calculated within each profile x model
stratum. The inferential test uses the resulting paired differences, thereby
avoiding treatment of all perturbation cases as independent observations.
Run numbers are not paired.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "preprocessing"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
LOG_DIR = BASE_DIR / "logs"

for directory in (RESULTS_DIR, FIGURES_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CODED_FILE = DATA_DIR / "llm_lead_intent_results_explanations_coded.csv"

ALPHA = 0.05
N_BOOTSTRAP = 10_000
RANDOM_SEED = 2026
EXPECTED_RUNS_PER_CASE = 5

# Adjust aliases here only if the CSV uses a different feature spelling.
FEATURE_ALIASES = {
    "company_size": "company_size",
    "company size": "company_size",
    "industry": "industry",
    "region": "region",
    "website_dwell_time": "website_dwell_time",
    "website dwell time": "website_dwell_time",
    "dwell_time": "website_dwell_time",
    "dwell time": "website_dwell_time",
    "demo_requests": "demo_requests",
    "demo requests": "demo_requests",
    "demo_request": "demo_requests",
    "email_response": "email_response",
    "email response": "email_response",
}

FEATURE_TYPE = {
    "company_size": "Firmographic",
    "industry": "Firmographic",
    "region": "Firmographic",
    "website_dwell_time": "Behavioral",
    "demo_requests": "Behavioral",
    "email_response": "Behavioral",
}

FEATURE_LABELS = {
    "company_size": "Company size",
    "industry": "Industry",
    "region": "Region",
    "website_dwell_time": "Website dwell time",
    "demo_requests": "Demo requests",
    "email_response": "Email response",
}

FEATURE_ORDER = [
    "company_size",
    "industry",
    "region",
    "website_dwell_time",
    "demo_requests",
    "email_response",
]
TYPE_ORDER = ["Firmographic", "Behavioral"]
COLORS = {"Firmographic": "#9ecae1", "Behavioral": "#08519c"}


# Logging
logger = logging.getLogger("h1_h3")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(
    LOG_DIR / "h1_h3_feature_alignment.log", mode="w", encoding="utf-8"
)
stream_handler = logging.StreamHandler()
for handler in (file_handler, stream_handler):
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def log(message: str = "") -> None:
    logger.info(message)


# Helpers
def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {CODED_FILE.name}: {missing}")


def normalize_feature(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower().map(FEATURE_ALIASES)
    if normalized.isna().any():
        unknown = sorted(series[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(
            f"Unknown feature labels: {unknown}. Add them to FEATURE_ALIASES."
        )
    return normalized


def derive_base_profile(profile_id: pd.Series) -> pd.Series:
    return profile_id.astype(str).str.split("_", n=1).str[0]


def percentile_bootstrap_mean_ci(
    values: pd.Series | np.ndarray,
    alpha: float = ALPHA,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_bootstrap, len(x)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def summarize_group(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, subset in data.groupby(group_column, sort=False):
        low, high = percentile_bootstrap_mean_ci(
            subset["alignment_share"], seed=RANDOM_SEED + len(rows)
        )
        rows.append(
            {
                group_column: group,
                "n_cases": len(subset),
                "n_explanations": int(subset["n_explanations"].sum()),
                "n_code3": int(subset["n_code3"].sum()),
                "mean_alignment_share": subset["alignment_share"].mean(),
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
                "n_complete_alignment_5_of_5": int(
                    subset["complete_alignment"].sum()
                ),
                "complete_alignment_rate": subset["complete_alignment"].mean(),
            }
        )
    return pd.DataFrame(rows)


def exact_sign_flip_test(differences: np.ndarray) -> tuple[float, int]:
    """Two-sided exact paired randomization test of mean difference = 0."""
    differences = np.asarray(differences, dtype=float)
    differences = differences[~np.isnan(differences)]
    n = len(differences)
    if n == 0:
        return np.nan, 0
    if n > 20:
        raise ValueError("Exact enumeration is restricted to at most 20 pairs.")

    observed = abs(differences.mean())
    permuted = np.empty(2**n, dtype=float)
    for i, signs in enumerate(itertools.product([-1.0, 1.0], repeat=n)):
        permuted[i] = abs(np.mean(differences * np.asarray(signs)))

    # Exact p-value; numerical tolerance protects equality comparisons.
    p_value = np.mean(permuted >= observed - 1e-12)
    return float(p_value), int(2**n)


def paired_bootstrap_difference_ci(
    paired: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Bootstrap paired profile-model strata and preserve within-stratum pairing."""
    differences = paired["difference_behavioral_minus_firmographic"].to_numpy(float)
    if len(differences) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        differences, size=(n_bootstrap, len(differences)), replace=True
    ).mean(axis=1)
    return tuple(np.quantile(draws, [alpha / 2, 1 - alpha / 2]))


# Load and prepare case-level alignment data
if not CODED_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {CODED_FILE}")

coding = pd.read_csv(CODED_FILE)
require_columns(coding, ["profile_id", "model", "perturbed_feature", "final_code"])

coding["feature"] = normalize_feature(coding["perturbed_feature"])
coding["feature_type"] = coding["feature"].map(FEATURE_TYPE)
coding["base_profile"] = derive_base_profile(coding["profile_id"])
coding["final_code"] = pd.to_numeric(coding["final_code"], errors="raise").astype(int)

if not coding["final_code"].isin([0, 1, 2, 3]).all():
    raise ValueError("final_code must contain only 0, 1, 2, or 3.")

coding["aligned"] = coding["final_code"].eq(3).astype(int)

# One case = one perturbation condition x model. profile_id retains low/high case.
case = (
    coding.groupby(
        ["profile_id", "base_profile", "model", "feature", "feature_type"],
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
case["complete_alignment"] = (
    case["n_code3"] == case["n_explanations"]
).astype(int)

if not case["n_explanations"].eq(EXPECTED_RUNS_PER_CASE).all():
    log("WARNING: Not every perturbation case contains exactly five explanations.")

observed_features = set(case["feature"].unique())
expected_features = set(FEATURE_TYPE)
if observed_features != expected_features:
    raise ValueError(
        "Feature set does not match the expected six features. "
        f"Observed={sorted(observed_features)}; expected={sorted(expected_features)}"
    )

# Descriptive results by feature and feature type
feature_summary = summarize_group(case, "feature")
feature_summary["feature_type"] = feature_summary["feature"].map(FEATURE_TYPE)
feature_summary["feature_label"] = feature_summary["feature"].map(FEATURE_LABELS)
feature_summary["order"] = feature_summary["feature"].map(
    {feature: i for i, feature in enumerate(FEATURE_ORDER)}
)
feature_summary = feature_summary.sort_values("order").drop(columns="order")

type_summary = summarize_group(case, "feature_type")
type_summary["order"] = type_summary["feature_type"].map(
    {feature_type: i for i, feature_type in enumerate(TYPE_ORDER)}
)
type_summary = type_summary.sort_values("order").drop(columns="order")

# Full Code 0-3 distribution by feature and type for transparent supplementary use.
# Complete Code 0–3 distribution by feature, including zero-count combinations.
feature_code_grid = pd.MultiIndex.from_product(
    [FEATURE_ORDER, [0, 1, 2, 3]],
    names=["feature", "final_code"],
).to_frame(index=False)

observed_feature_codes = (
    coding.groupby(["feature", "final_code"])
    .size()
    .rename("count")
    .reset_index()
)

code_distribution_feature = feature_code_grid.merge(
    observed_feature_codes,
    on=["feature", "final_code"],
    how="left",
)
code_distribution_feature["count"] = (
    code_distribution_feature["count"].fillna(0).astype(int)
)
code_distribution_feature["feature_type"] = (
    code_distribution_feature["feature"].map(FEATURE_TYPE)
)
code_distribution_feature["feature_label"] = (
    code_distribution_feature["feature"].map(FEATURE_LABELS)
)
code_distribution_feature["share_within_feature"] = (
    code_distribution_feature["count"]
    / code_distribution_feature.groupby("feature")["count"].transform("sum")
)
code_distribution_feature["feature_order"] = (
    code_distribution_feature["feature"].map(
        {feature: i for i, feature in enumerate(FEATURE_ORDER)}
    )
)
code_distribution_feature = (
    code_distribution_feature
    .sort_values(["feature_order", "final_code"])
    .drop(columns="feature_order")
    .reset_index(drop=True)
)

# Complete Code 0–3 distribution by feature type, also retaining zero cells.
type_code_grid = pd.MultiIndex.from_product(
    [TYPE_ORDER, [0, 1, 2, 3]],
    names=["feature_type", "final_code"],
).to_frame(index=False)

observed_type_codes = (
    coding.groupby(["feature_type", "final_code"])
    .size()
    .rename("count")
    .reset_index()
)

code_distribution_type = type_code_grid.merge(
    observed_type_codes,
    on=["feature_type", "final_code"],
    how="left",
)
code_distribution_type["count"] = (
    code_distribution_type["count"].fillna(0).astype(int)
)
code_distribution_type["share_within_type"] = (
    code_distribution_type["count"]
    / code_distribution_type.groupby("feature_type")["count"].transform("sum")
)

# Paired feature-type comparison within profile x model strata
stratum_type = (
    case.groupby(["base_profile", "model", "feature_type"], as_index=False)
    .agg(
        mean_alignment_share=("alignment_share", "mean"),
        n_cases=("alignment_share", "size"),
    )
)

paired = stratum_type.pivot(
    index=["base_profile", "model"],
    columns="feature_type",
    values="mean_alignment_share",
).reset_index()

missing_types = [feature_type for feature_type in TYPE_ORDER if feature_type not in paired]
if missing_types:
    raise ValueError(f"Missing feature types in paired comparison: {missing_types}")
if paired[TYPE_ORDER].isna().any().any():
    raise ValueError("At least one profile-model stratum lacks one feature type.")

paired["difference_behavioral_minus_firmographic"] = (
    paired["Behavioral"] - paired["Firmographic"]
)

observed_difference = paired["difference_behavioral_minus_firmographic"].mean()
p_value, n_permutations = exact_sign_flip_test(
    paired["difference_behavioral_minus_firmographic"].to_numpy()
)
diff_ci_low, diff_ci_high = paired_bootstrap_difference_ci(paired)

inferential_result = pd.DataFrame(
    [
        {
            "comparison": "Behavioral minus firmographic",
            "n_paired_profile_model_strata": len(paired),
            "mean_paired_difference": observed_difference,
            "paired_bootstrap_95ci_low": diff_ci_low,
            "paired_bootstrap_95ci_high": diff_ci_high,
            "test": "Two-sided exact sign-flip permutation test",
            "n_permutations": n_permutations,
            "p_value": p_value,
            "alpha": ALPHA,
            "reject_equal_alignment_null": bool(p_value < ALPHA),
        }
    ]
)

# Save tables
case.to_csv(RESULTS_DIR / "h1_h3_case_level_alignment.csv", index=False)
feature_summary.to_csv(RESULTS_DIR / "h1_h3_alignment_by_feature.csv", index=False)
type_summary.to_csv(RESULTS_DIR / "h1_h3_alignment_by_feature_type.csv", index=False)
paired.to_csv(RESULTS_DIR / "h1_h3_paired_profile_model_strata.csv", index=False)
inferential_result.to_csv(RESULTS_DIR / "h1_h3_feature_type_test.csv", index=False)
code_distribution_feature.to_csv(
    RESULTS_DIR / "h1_h3_code_distribution_by_feature.csv", index=False
)
code_distribution_type.to_csv(
    RESULTS_DIR / "h1_h3_code_distribution_by_feature_type.csv", index=False
)

# Log concise results
log("Complementary H1 + H3 Analysis")
log("================================")
log("Outcome: case-level share of explanations assigned Code 3.")
log("Scope: all perturbation cases; run numbers are not paired.")
log("Primary comparison: behavioral vs. firmographic features.")
log("Pairing unit: profile x model stratum.")
log("")
log("Alignment by feature:")
log(
    feature_summary[
        [
            "feature_label",
            "feature_type",
            "n_cases",
            "mean_alignment_share",
            "bootstrap_95ci_low",
            "bootstrap_95ci_high",
            "complete_alignment_rate",
        ]
    ].to_string(index=False)
)
log("\nAlignment by feature type:")
log(type_summary.to_string(index=False))
log("\nPaired feature-type test:")
log(inferential_result.to_string(index=False))

# Figure 1: alignment by individual feature
plot_feature = feature_summary.copy()
y = plot_feature["mean_alignment_share"].to_numpy(float)
low = plot_feature["bootstrap_95ci_low"].to_numpy(float)
high = plot_feature["bootstrap_95ci_high"].to_numpy(float)
yerr = np.vstack([np.maximum(0, y - low), np.maximum(0, high - y)])
colors = plot_feature["feature_type"].map(COLORS).tolist()

fig, ax = plt.subplots(figsize=(9, 5.8))
x = np.arange(len(plot_feature))
bars = ax.bar(
    x,
    y,
    yerr=yerr,
    capsize=5,
    color=colors,
    edgecolor="white",
    linewidth=0.8,
    error_kw={"ecolor": "#222222", "elinewidth": 1.3, "capthick": 1.3},
)
for bar, value, upper in zip(bars, y, high):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        min(upper + 0.025, 1.035),
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
ax.axhline(1.0, color="#222222", linestyle=":", linewidth=1.6)
ax.set_ylim(0, 1.08)
ax.set_xticks(x)
ax.set_xticklabels(plot_feature["feature_label"], rotation=25, ha="right")
ax.set_ylabel("Mean share of explanations with Code 3")
ax.set_title("Explanation alignment by manipulated feature")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=COLORS[feature_type], label=feature_type)
    for feature_type in TYPE_ORDER
]
legend_handles.append(
    plt.Line2D([0], [0], color="#222222", linestyle=":", label="Perfect alignment")
)
ax.legend(handles=legend_handles, loc="upper right", frameon=True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.text(
    0.5,
    0.01,
    "Error bars show 95% case-bootstrap confidence intervals.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(
    FIGURES_DIR / "h1_h3_fig1_alignment_by_feature.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Figure 2: feature-type means plus paired profile-model strata
plot_type = type_summary.set_index("feature_type").loc[TYPE_ORDER].reset_index()
y = plot_type["mean_alignment_share"].to_numpy(float)
low = plot_type["bootstrap_95ci_low"].to_numpy(float)
high = plot_type["bootstrap_95ci_high"].to_numpy(float)
yerr = np.vstack([np.maximum(0, y - low), np.maximum(0, high - y)])

fig, ax = plt.subplots(figsize=(7.5, 5.8))
x = np.arange(2)

# Thin paired lines show all profile-model strata.
for _, row in paired.iterrows():
    ax.plot(
        x,
        [row["Firmographic"], row["Behavioral"]],
        color="#777777",
        alpha=0.32,
        linewidth=1.0,
        marker="o",
        markersize=3,
        zorder=1,
    )

bars = ax.bar(
    x,
    y,
    yerr=yerr,
    capsize=6,
    width=0.58,
    color=[COLORS[t] for t in TYPE_ORDER],
    alpha=0.82,
    edgecolor="white",
    linewidth=0.8,
    error_kw={"ecolor": "#111111", "elinewidth": 1.5, "capthick": 1.5},
    zorder=2,
)
for bar, value, upper in zip(bars, y, high):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        min(upper + 0.025, 1.035),
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        zorder=3,
    )
ax.axhline(1.0, color="#222222", linestyle=":", linewidth=1.6)
ax.set_ylim(0, 1.08)
ax.set_xticks(x)
ax.set_xticklabels(TYPE_ORDER)
ax.set_ylabel("Mean share of explanations with Code 3")
ax.set_title("Explanation alignment by feature type")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(
    0.5,
    0.035,
    f"Paired difference (behavioral − firmographic) = {observed_difference:.2f}; "
    f"exact two-sided p = {p_value:.3f}",
    ha="center",
    fontsize=9,
)
fig.text(
    0.5,
    0.012,
    "Bars: overall means with 95% case-bootstrap CIs; lines: profile–model strata.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(
    FIGURES_DIR / "h1_h3_fig2_alignment_by_feature_type.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Figure 3: complete Code 0–3 distribution by manipulated feature
code_colors = {
    0: "#d9d9d9",
    1: "#fdae6b",
    2: "#e6550d",
    3: "#3182bd",
}
code_labels = {
    0: "Code 0: not specified",
    1: "Code 1: neutral",
    2: "Code 2: refuted",
    3: "Code 3: explanatory alignment",
}

code_pivot = (
    code_distribution_feature
    .pivot(index="feature", columns="final_code", values="share_within_feature")
    .reindex(FEATURE_ORDER)
    .reindex(columns=[0, 1, 2, 3], fill_value=0)
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(9, 5.8))
left = np.zeros(len(code_pivot), dtype=float)
y_labels = [FEATURE_LABELS[feature] for feature in code_pivot.index]

for code in [0, 1, 2, 3]:
    values = code_pivot[code].to_numpy(dtype=float)
    bars = ax.barh(
        y_labels,
        values,
        left=left,
        color=code_colors[code],
        edgecolor="white",
        linewidth=0.7,
        label=code_labels[code],
    )

    # Suppress labels for very small segments to preserve readability.
    for bar, value, start in zip(bars, values, left):
        if value >= 0.08:
            text_color = "white" if code in (2, 3) else "#222222"
            ax.text(
                start + value / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=text_color,
            )
    left += values

ax.set_xlim(0, 1)
ax.set_xlabel("Share of explanations")
ax.set_ylabel("")
ax.set_title("Explanation-code distribution by manipulated feature")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
ax.invert_yaxis()
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=2,
    frameon=False,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig(
    FIGURES_DIR / "h1_h3_fig3_code_distribution_by_feature.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

log("\nComplete Code 0–3 distribution by feature:")
log(
    code_distribution_feature[
        ["feature_label", "feature_type", "final_code", "count", "share_within_feature"]
    ].to_string(index=False)
)
