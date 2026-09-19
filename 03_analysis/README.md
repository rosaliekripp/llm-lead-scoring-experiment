# Analysis

This directory contains the preprocessing workflow and the statistical analyses for hypotheses H1–H4. It also contains exploratory cross-hypothesis analyses.

Run the preprocessing scripts before executing the hypothesis-specific analyses. For the analyses reported in the study, use the archived experiment results rather than newly generated model outputs.

## Hypotheses

- **H1 — Feature Sensitivity:** Changes in behavior-based input features are more likely to change the lead score than changes in static company data.
- **H2 — Stability:** Despite identical input, repeated model outputs exhibit variation in lead scores and their associated explanations.
- **H3 — Explanation Alignment:** When an input change influences the lead score, the manipulated feature is not always cited as a contributing factor in the model's self-explanation.
- **H4 — Model Differences:** Lead scores, output stability, perturbation responses, and associated explanations differ between language models when they receive identical input data.

## Directory and Hypothesis Mapping

| Directory | Hypothesis | Purpose |
|---|---|---|
| `preprocessing/` | H1–H4 | Cleans the experiment results and prepares the explanation codes used in the statistical analyses. |
| `h1/` | H1 | Tests whether behavioral features produce classification changes more frequently than firmographic features. |
| `h1_x/` | Exploratory | Examines relationships between H1 and H2, H3, and H4. The `_x` suffix denotes cross-hypothesis analyses and does not represent an additional confirmatory hypothesis. |
| `h2/` | H2 | Assesses the repeatability of lead scores and coded explanation attributions across identical runs. |
| `h3/` | H3 | Evaluates whether the experimentally manipulated feature is cited as a contributing explanatory factor. |
| `h4/` | H4 | Compares classifications, perturbation responses, stability, and explanation behavior across models. |

See the individual folders and script headers for exact input files, output files, and execution details.

## Preprocessing

The preprocessing workflow converts the archived experiment outputs into analysis-ready datasets.

Run the scripts in this order:

```bash
python3 03_analysis/preprocessing/clean_results.py
python3 03_analysis/preprocessing/explanation_coding_scheme.py
```

`clean_results.py` standardizes the model responses, recodes formally invalid label representations where required, removes operational columns not needed for analysis, and produces the cleaned dataset.

`explanation_coding_scheme.py` assigns a suggested explanation code to each perturbation response. The coding is anchored exclusively to the feature manipulated in the corresponding condition.

| Code | Category | Definition |
|---:|---|---|
| 0 | Not specified | The manipulated feature is not mentioned or clearly described. |
| 1 | Specified, neutral | The feature is mentioned but is not assigned an effect on the classification. |
| 2 | Specified, refuted | The feature is acknowledged but explicitly outweighed by other factors. |
| 3 | Supportive, explanatory | The feature is cited as contributing to the classification. |

Baseline observations are excluded from explanation coding because no feature is manipulated. Algorithmic pre-coding serves as a review aid; the final codes are based on the documented coding rules and manual review.

## Methods

### H1: Feature Sensitivity

For each profile, feature, model, and run, the low and high perturbation conditions form a matched pair. Feature sensitivity is defined as whether the binary lead score differs between these conditions:

```text
ΔScore = Score_High − Score_Low
Sensitivity = 1 if |ΔScore| = 1, otherwise 0
```

The primary analysis uses a population-averaged logistic generalized estimating equation (GEE). Feature type is the predictor of interest, model is included as a categorical fixed effect, and observations are clustered by profile and manipulated feature.

Supplementary analyses examine:

- changes relative to the corresponding baseline;
- directionality of observed score changes;
- sensitivity by individual feature;
- raw and model-adjusted probabilities; and
- leave-one-profile-out robustness.

### H2: Stability

Stability is assessed across the five repeated runs for each identical model-input condition.

Lead score and explanation attribution stability are analyzed separately using:

- Krippendorff's alpha for nominal data;
- complete agreement;
- modal agreement share; and
- case-level instability.

Lead score stability includes baseline and perturbation conditions. Explanation stability includes perturbation conditions only because baseline cases have no manipulated feature and therefore no explanation code.

The repeated model outputs are interpreted as repeated realizations under identical conditions, not as independent human raters.

### H3: Explanation Alignment

An explanation is classified as aligned when the manipulated feature receives **Code 3**, meaning that the feature is cited as contributing to the classification.

The primary analysis focuses on perturbation cases whose majority classification differs from the corresponding baseline majority classification. For each case, alignment is calculated across the five repeated runs:

```text
Alignment share = Number of Code 3 explanations / 5
```

Complete alignment requires Code 3 in all five runs.

A broader supplementary analysis includes all perturbation cases, irrespective of whether the classification changes relative to baseline. H3 is evaluated descriptively using alignment shares, complete-alignment rates, and confidence intervals.

### H4: Model Differences

H4 treats the outputs of the three models as matched and repeated measurements because every model receives the same experimental inputs.

The analyses include:

- Bayesian logistic mixed models for binary classifications, perturbation responses, and explanation alignment;
- Friedman tests for omnibus comparisons of model-level stability;
- paired Wilcoxon tests for follow-up stability comparisons; and
- exploratory one-versus-rest models for the nominal explanation-code categories.

Pairwise model comparisons are conducted only after a statistically significant omnibus model effect. Holm correction is applied within predefined multiple-testing families.

## Statistical Conventions

- The significance level is set to `α = .05`.
- Effect estimates are reported with confidence intervals where applicable.
- Robust standard errors are used for the GEE analyses.
- Bootstrap confidence intervals are used for Krippendorff's alpha and selected case-level estimates.
- Holm-adjusted p-values are used for predefined families of multiple comparisons.
- H2 and H3 are primarily evaluated using descriptive agreement and alignment measures rather than tests against arbitrary thresholds.
- Confirmatory, supplementary, and exploratory analyses are identified separately.

## Execution Order

Run the workflow from the repository root in the following order:

1. Clean the archived experiment results.
2. Generate the algorithmic explanation-code suggestions.
3. Use the reviewed final explanation codes for the statistical analyses.
4. Run the analyses in `h1/`.
5. Run the analyses in `h2/`.
6. Run the analyses in `h3/`.
7. Run the analyses in `h4/`.
8. Run the exploratory analyses in `h1_x/`.

The scripts within each hypothesis directory should be executed according to their numbering, filenames, or script-level documentation.

## Reproducibility Notes

The analysis scripts are designed to reproduce the reported results from the archived model outputs. A fresh API run may produce different responses because model endpoints, provider infrastructure, and serving behavior can change over time.

Do not overwrite the archived raw or processed datasets when conducting a new replication. Store regenerated outputs separately and document the repository commit, Python version, dependency versions, and any deviations from the original workflow.