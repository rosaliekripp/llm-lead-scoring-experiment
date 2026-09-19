"""H1 feature-sensitivity analysis.

Primary estimand
-----------------
Sensitivity = 1 when the matched Low-High pair changes the binary score.
Scores are coded 0 = Low Intent and 1 = High Intent.

Primary inference
-----------------
Population-averaged logistic GEE:
    sensitivity ~ feature_type + model
Cluster: reference profile x manipulated feature.
Each cluster contains 3 models x 5 repeated runs = 15 observations.

Supplementary analyses
----------------------
1. Raw sensitivity rates and signed Low-High changes.
2. Baseline-anchored changes, restricted to directionally change-eligible cases.
3. Directional consistency among perturbations that changed vs baseline.
4. Exploratory feature-level rates (descriptive).
5. Robustness checks using an independence working correlation and a
   profile-level leave-one-out analysis.

Outputs
-------
Three CSV result tables and at most five PNG figures are written to results/
and figures/. The script intentionally avoids exploratory pairwise Wald tests:
with only six features and possible boundary rates, those tests are unstable
and do not add reliable evidence beyond the prespecified H1 contrast.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import confint_proportions_2indep

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 40)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "preprocessing" / "llm_lead_intent_results_clean.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
LOG_DIR = BASE_DIR / "logs"
for directory in (RESULTS_DIR, FIGURES_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class Tee:
    """Write console output to the terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


log_file = open(LOG_DIR / "h1_feature_sensitivity.log", "w", encoding="utf-8")
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)

BEHAVIORAL = ["website_dwell_time", "demo_requests", "email_response"]
FIRMOGRAPHIC = ["company_size", "industry", "region"]
FEATURES = BEHAVIORAL + FIRMOGRAPHIC
SCORE_MAP = {"Low Intent": 0, "High Intent": 1}


def wilson(successes, total, alpha=0.05):
    """Return a Wilson confidence interval; return missing values for n=0."""
    if total == 0:
        return np.nan, np.nan
    low, high = proportion_confint(successes, total, alpha=alpha, method="wilson")
    return float(low), float(high)


def get_feature(perturbation):
    """Extract the manipulated feature from the perturbation label."""
    for feature in FEATURES:
        if str(perturbation).startswith(feature):
            return feature
    return "BASE"


def get_direction(perturbation):
    """Extract the perturbation direction from the perturbation label."""
    perturbation = str(perturbation)
    for direction in ("non-icp", "icp", "high", "low"):
        if perturbation.endswith(direction):
            return direction
    return "base"


def rate_table(data, group_cols, outcome):
    """Calculate counts, rates, and descriptive Wilson intervals."""
    table = (data.groupby(group_cols, observed=True)[outcome]
             .agg(successes="sum", total="count").reset_index())
    table["rate"] = table["successes"] / table["total"]
    intervals = [wilson(int(k), int(n)) for k, n in zip(table.successes, table.total)]
    table[["ci_low", "ci_high"]] = pd.DataFrame(intervals, index=table.index)
    table["ci_type"] = "Wilson descriptive; does not adjust for clustering"
    return table


def fit_gee(data, covariance="exchangeable"):
    """Fit the prespecified H1 GEE with model as a categorical fixed effect."""
    cov_struct = Exchangeable() if covariance == "exchangeable" else Independence()
    return GEE.from_formula(
        "sensitivity ~ fe_behavioral + C(model)",
        groups="cluster_pf",
        data=data,
        family=Binomial(),
        cov_struct=cov_struct,
    ).fit()


def primary_effect_row(result, analysis):
    """Extract the adjusted feature-type odds ratio from a fitted GEE."""
    term = "fe_behavioral"
    beta = float(result.params[term])
    se = float(result.bse[term])
    return {
        "analysis": analysis,
        "term": "Behavioral vs firmographic",
        "log_odds": beta,
        "robust_se": se,
        "odds_ratio": float(np.exp(beta)),
        "ci_low": float(np.exp(beta - 1.96 * se)),
        "ci_high": float(np.exp(beta + 1.96 * se)),
        "p_value": float(result.pvalues[term]),
        "n_observations": int(result.nobs),
        "n_clusters": int(result.model.num_group),
        "note": "GEE robust Wald inference for the single prespecified H1 contrast",
    }


def adjusted_predictions(result, data):
    """Standardize predictions over the observed model distribution."""
    rows = []
    model_levels = sorted(data["model"].dropna().unique())
    for feature_type, indicator in (("firmographic", 0), ("behavioral", 1)):
        new = pd.DataFrame({"fe_behavioral": indicator, "model": model_levels})
        probabilities = np.asarray(result.predict(new), dtype=float)
        rows.append({
            "feature_type": feature_type,
            "adjusted_probability": probabilities.mean(),
        })
    output = pd.DataFrame(rows)
    difference = (output.loc[output.feature_type == "behavioral", "adjusted_probability"].iloc[0]
                  - output.loc[output.feature_type == "firmographic", "adjusted_probability"].iloc[0])
    return output, float(difference)


# 1. Data preparation and matched Low-High pairs
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Input data not found: {DATA_PATH}. Place the cleaned CSV in preprocessing/."
    )

df = pd.read_csv(DATA_PATH)
required = {"intent", "profile_id", "model", "run"}
missing = required.difference(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["score"] = df["intent"].astype(str).str.strip().map(SCORE_MAP)
if df["score"].isna().any():
    invalid = sorted(df.loc[df.score.isna(), "intent"].astype(str).unique())
    raise ValueError(f"Unrecognized intent values: {invalid}")

df["base_profile"] = df["profile_id"].astype(str).str.split("_").str[0]
df["perturbation"] = df["profile_id"].astype(str).str.split("_", n=1).str[1]
df["feature"] = df["perturbation"].apply(get_feature)
df["direction"] = df["perturbation"].apply(get_direction)
df["pole"] = df["direction"].map(
    {"high": "high", "icp": "high", "low": "low", "non-icp": "low"}
)
df["feature_type"] = np.select(
    [df.feature.isin(BEHAVIORAL), df.feature.isin(FIRMOGRAPHIC)],
    ["behavioral", "firmographic"],
    default="base",
)

pert = df[df.feature != "BASE"].copy()
pairs = (pert.pivot_table(
    index=["base_profile", "feature", "feature_type", "model", "run"],
    columns="pole", values="score", aggfunc="first")
    .reset_index().dropna(subset=["low", "high"]))
pairs["delta"] = pairs["high"] - pairs["low"]
pairs["sensitivity"] = pairs["delta"].abs().eq(1).astype(int)
pairs["fe_behavioral"] = pairs.feature_type.eq("behavioral").astype(int)
# The cluster is the substantive manipulation, shared across models and runs.
pairs["cluster_pf"] = pairs["base_profile"].astype(str) + "__" + pairs["feature"].astype(str)

expected_pairs = (df.base_profile.nunique() * len(FEATURES)
                  * df.model.nunique() * df.run.nunique())
if len(pairs) != expected_pairs:
    print(f"WARNING: obtained {len(pairs)} matched pairs; expected {expected_pairs}.")

print("\nPRIMARY DATASET")
print(f"Matched Low-High pairs: {len(pairs)}")
print(f"Profile-feature clusters: {pairs.cluster_pf.nunique()}")


# 2. Primary confirmatory H1 analysis
print("\nPRIMARY CONFIRMATORY ANALYSIS")
primary_gee = fit_gee(pairs, covariance="exchangeable")
print(primary_gee.summary())
primary_results = pd.DataFrame([
    primary_effect_row(primary_gee, "Primary GEE, exchangeable correlation")
])

adjusted, adjusted_rd = adjusted_predictions(primary_gee, pairs)
adjusted["analysis"] = "Primary model-standardized probabilities"
adjusted["adjusted_risk_difference_behavioral_minus_firmographic"] = adjusted_rd

# Descriptive rates support interpretation but are not the inferential test.
type_rates = rate_table(pairs, ["feature_type"], "sensitivity")
type_rates["analysis"] = "Raw descriptive sensitivity rate"
model_type_rates = rate_table(pairs, ["model", "feature_type"], "sensitivity")
model_type_rates["analysis"] = "Raw descriptive sensitivity rate by model"


# 3. Supplementary signed-change analysis
print("\nSUPPLEMENTARY SIGNED LOW-HIGH CHANGES")
signed_counts = (pairs.groupby(["feature_type", "feature", "delta"], observed=True)
                 .size().rename("count").reset_index())
signed_counts["share_within_feature"] = (
    signed_counts["count"]
    / signed_counts.groupby("feature", observed=True)["count"].transform("sum")
)
signed_counts["analysis"] = "Signed Low-High delta distribution"


# 4. Supplementary baseline-anchored and directional analyses
print("\nSUPPLEMENTARY BASELINE-ANCHORED ANALYSIS")
base = (df[df.feature == "BASE"][["base_profile", "model", "run", "score"]]
        .rename(columns={"score": "base_score"}))
anchored = pert.merge(base, on=["base_profile", "model", "run"], how="left")
if anchored.base_score.isna().any():
    raise ValueError("At least one perturbation has no matching baseline observation.")
anchored["delta_base"] = anchored["score"] - anchored["base_score"]
anchored["changed"] = anchored["delta_base"].abs().eq(1).astype(int)
anchored["expected_dir"] = np.where(anchored.pole.eq("high"), 1, -1)
# Baseline 0 is Low Intent; only an upward change is possible.
# Baseline 1 is High Intent; only a downward change is possible.
anchored["change_eligible"] = np.where(
    anchored.expected_dir.eq(1), anchored.base_score.eq(0), anchored.base_score.eq(1)
)
anchored["fe_behavioral"] = anchored.feature_type.eq("behavioral").astype(int)
anchored["cluster_pf"] = anchored.base_profile.astype(str) + "__" + anchored.feature.astype(str)

eligible = anchored[anchored.change_eligible].copy()
baseline_gee = GEE.from_formula(
    "changed ~ fe_behavioral + C(model)",
    groups="cluster_pf", data=eligible, family=Binomial(),
    cov_struct=Exchangeable(),
).fit()
baseline_results = pd.DataFrame([
    primary_effect_row(baseline_gee, "Supplementary baseline-anchored GEE; eligible cases")
])
eligible_rates = rate_table(eligible, ["feature_type"], "changed")
eligible_rates["analysis"] = "Baseline-anchored change rate; eligible cases"

moved = anchored[anchored.changed.eq(1)].copy()
moved["as_expected"] = np.sign(moved.delta_base).eq(moved.expected_dir).astype(int)
directional_rates = rate_table(
    moved, ["feature_type", "feature"], "as_expected"
) if len(moved) else pd.DataFrame(columns=[
    "feature_type", "feature", "successes", "total", "rate", "ci_low", "ci_high", "ci_type"
])
directional_rates["analysis"] = "Directional consistency among baseline changes"


# 5. Exploratory feature-level analysis (descriptive)
print("\nEXPLORATORY FEATURE-LEVEL ANALYSIS")
feature_rates = rate_table(pairs, ["feature_type", "feature"], "sensitivity")
feature_rates["analysis"] = "Exploratory feature-level sensitivity"
feature_rates = feature_rates.sort_values("rate", ascending=False)
print(feature_rates.to_string(index=False))


# 6. Robustness analyses
print("\nROBUSTNESS ANALYSES")
independence_gee = fit_gee(pairs, covariance="independence")
robustness_rows = [
    primary_effect_row(independence_gee, "Robustness GEE, independence correlation")
]
# Leave one profile out to detect dependence on a single synthetic scenario.
for profile in sorted(pairs.base_profile.unique()):
    subset = pairs[pairs.base_profile.ne(profile)].copy()
    result = fit_gee(subset, covariance="exchangeable")
    robustness_rows.append(primary_effect_row(
        result, f"Leave-one-profile-out; omitted {profile}"
    ))
robustness_results = pd.DataFrame(robustness_rows)


# 7. Save result tables
confirmatory_table = pd.concat([primary_results, baseline_results], ignore_index=True)
descriptive_table = pd.concat([
    type_rates, model_type_rates, adjusted, eligible_rates,
    directional_rates, feature_rates, signed_counts,
], ignore_index=True, sort=False)

confirmatory_table.to_csv(RESULTS_DIR / "h1_confirmatory_and_supplementary_models.csv", index=False)
descriptive_table.to_csv(RESULTS_DIR / "h1_descriptive_results.csv", index=False)
robustness_results.to_csv(RESULTS_DIR / "h1_robustness_results.csv", index=False)
pairs.to_csv(RESULTS_DIR / "h1_matched_pair_analysis_data.csv", index=False)


# 8. Figures
sns.set_theme(style="whitegrid", context="notebook")
palette = {
    "behavioral": "#2C7FB8",
    "firmographic": "#D95F0E",
}
# Descriptive feature-type sensitivity rates.
# These intervals do not account for clustering and therefore support
# interpretation but do not replace the primary GEE inference.
rates = rate_table(
    pairs,
    ["feature_type"],
    "sensitivity",
).set_index("feature_type")
# Raw, unadjusted risk difference:
# behavioral sensitivity rate minus firmographic sensitivity rate.
behavioral_successes = int(rates.loc["behavioral", "successes"])
behavioral_total = int(rates.loc["behavioral", "total"])
firmographic_successes = int(rates.loc["firmographic", "successes"])
firmographic_total = int(rates.loc["firmographic", "total"])
rd = (
    rates.loc["behavioral", "rate"]
    - rates.loc["firmographic", "rate"]
)
rd_low, rd_high = confint_proportions_2indep(
    behavioral_successes,
    behavioral_total,
    firmographic_successes,
    firmographic_total,
    method="newcomb",
    compare="diff",
)
# Descriptive feature-level rates.
feat_rates = rate_table(
    pairs,
    ["feature_type", "feature"],
    "sensitivity",
).set_index(["feature_type", "feature"])
# Signed Low-High delta counts.
delta_tab = pd.crosstab(
    pairs["feature"],
    pairs["delta"],
).reindex(columns=[-1, 0, 1], fill_value=0)

# Figure 1: baseline classification context.
# Each reference profile has 3 models x 5 runs = 15 baseline observations.
base_rows = df[df["feature"] == "BASE"].copy()
counts = (
    base_rows
    .groupby(["base_profile", "intent"], observed=True)
    .size()
    .reset_index(name="n")
)
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(
    data=counts,
    x="base_profile",
    y="n",
    hue="intent",
    hue_order=["Low Intent", "High Intent"],
    palette={
        "High Intent": "#31A354",
        "Low Intent": "#DE2D26",
    },
    ax=ax,
)
ax.set(
    title="Baseline intent classification per reference profile",
    xlabel="Reference profile",
    ylabel="Number of model-run observations",
)
ax.legend(
    title="Intent classification",
    loc="upper right",
)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "h1_fig1_baseline_intent.png",
    dpi=200,
)
plt.close(fig)

# Figure 2: primary descriptive H1 result.
# Wilson intervals and the Newcombe risk-difference interval are descriptive
# and do not account for clustering. Confirmatory inference comes from the GEE.
fig, ax = plt.subplots(figsize=(6.5, 5.2))
r = (
    rates
    .reset_index()
    .sort_values("feature_type")
    .reset_index(drop=True)
)
x_positions = np.arange(len(r))
ax.bar(
    x_positions,
    r["rate"],
    color=[palette[feature_type] for feature_type in r["feature_type"]],
    width=0.55,
)
ax.errorbar(
    x_positions,
    r["rate"],
    yerr=[
        r["rate"] - r["ci_low"],
        r["ci_high"] - r["rate"],
    ],
    fmt="none",
    ecolor="black",
    elinewidth=1.4,
    capsize=8,
)
for i, row in r.iterrows():
    label_y = min(row["ci_high"] + 0.035, 0.97)

    ax.text(
        i,
        label_y,
        f"{row['rate']:.1%}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
ax.annotate(
    (
        f"Risk difference: {rd * 100:.1f} pp\n"
        f"95% CI [{rd_low * 100:.1f}, {rd_high * 100:.1f}]"
    ),
    xy=(0.5, 0.80),
    xycoords="axes fraction",
    ha="center",
    fontsize=10,
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "edgecolor": "grey",
        "alpha": 0.95,
    },
)
ax.set_xticks(
    x_positions,
    [feature_type.title() for feature_type in r["feature_type"]],
)
ax.set(
    title="Sensitivity rate by feature type",
    xlabel="Feature type",
    ylabel="P(|ΔScore| = 1)",
    ylim=(0, 1),
)
fig.tight_layout()

plt.close(fig)

# Figure 3: exploratory feature-level sensitivity rates.
# Wilson intervals are descriptive and do not account for clustering.
fr = (
    feat_rates
    .reset_index()
    .sort_values("rate")
    .reset_index(drop=True)
)
fig, ax = plt.subplots(figsize=(7.5, 5))
for i, row in enumerate(fr.itertuples(index=False)):
    ax.plot(
        [row.ci_low, row.ci_high],
        [i, i],
        color=palette[row.feature_type],
        linewidth=2.5,
        solid_capstyle="round",
    )
    ax.plot(
        row.rate,
        i,
        marker="o",
        linestyle="none",
        color=palette[row.feature_type],
        markersize=9,
    )
ax.set_yticks(range(len(fr)))
ax.set_yticklabels(fr["feature"])
legend_handles = [
    plt.Line2D(
        [],
        [],
        color=color,
        marker="o",
        linestyle="-",
        linewidth=2.5,
        markersize=7,
        label=feature_type.title(),
    )
    for feature_type, color in palette.items()
]
ax.legend(
    handles=legend_handles,
    title="Feature type",
    loc="lower right",
)
ax.set(
    title="Sensitivity rate per feature",
    xlabel="P(|ΔScore| = 1)",
    ylabel="Feature",
    xlim=(0, 1),
)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "h1_fig3_feature_ranking.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Figure 4: composition of signed Low-High score changes per feature.
share = (
    delta_tab
    .div(delta_tab.sum(axis=1), axis=0)
    .reindex(columns=[-1, 0, 1], fill_value=0.0)
)
# Sort features by the share of positive Low-High changes.
feature_order = share[1].sort_values().index
share = share.loc[feature_order]
fig, ax = plt.subplots(figsize=(8, 5))
left = np.zeros(len(share))
delta_styles = [
    (-1, "#D95F02", "Decrease (-1)"),
    (0, "#BDBDBD", "No change (0)"),
    (1, "#1B7837", "Increase (+1)"),
]
for delta_value, color, label in delta_styles:
    ax.barh(
        share.index,
        share[delta_value],
        left=left,
        color=color,
        label=label,
    )
    left += share[delta_value].to_numpy()
ax.set(
    title="Composition of signed score changes per feature",
    xlabel="Share of matched Low-High pairs",
    ylabel="Feature",
    xlim=(0, 1),
)
ax.legend(
    title="ΔScore",
    loc="lower right",
)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "h1_fig4_signed_delta_composition.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Figure 5: exploratory sensitivity pattern by reference profile and feature.
# Values are averaged across models and runs.
profile_feature_rates = pairs.pivot_table(
    index="base_profile",
    columns="feature",
    values="sensitivity",
    aggfunc="mean",
)
# Use the predefined feature order for easier interpretation.
ordered_features = [
    feature
    for feature in FEATURES
    if feature in profile_feature_rates.columns
]
profile_feature_rates = profile_feature_rates.reindex(
    columns=ordered_features
)
fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(
    profile_feature_rates,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Sensitivity rate"},
    ax=ax,
)
ax.set(
    title="Exploratory sensitivity by reference profile and feature",
    xlabel="Feature",
    ylabel="Reference profile",
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=35,
    ha="right",
)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "h1_fig5_profile_feature_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

sys.stdout = original_stdout
log_file.close()
