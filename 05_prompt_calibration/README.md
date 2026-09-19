# Prompt Calibration

## Purpose

Before the main experiment, a prompt calibration study was conducted to select a single prompt configuration for the lead scoring task. The objective was to identify a prompt that reliably produced valid JSON responses, stable binary classifications, and concise self-explanations across heterogeneous lead profiles.

The calibration was not designed to compare model performance or optimize the prompt separately for each model. The selected prompt was frozen and subsequently used unchanged across all three models and all conditions in the main experiment.

## Tested Variants

Five prompt variants (`V1`–`V5`) were evaluated. The variants differed in instruction specificity and in the placement of the task description between the system and user prompts.

Each variant was tested on three heterogeneous baseline profiles from the experimental dataset:

- case ID 1;
- case ID 14; and
- case ID 27.

The selected profiles represented different combinations of firmographic characteristics and behavioral engagement. Each prompt-profile combination was repeated five times using `meta-llama-3.1-8b-instruct`.

```text
5 prompt variants × 3 profiles × 5 repeated runs = 75 model calls
```

The same model and parameter configuration were used for all calibration calls.

## Selection Criteria

The prompt variants were assessed descriptively using the following criteria:

- **Classification stability** — consistency of the binary intent label across the five repeated runs for the same prompt and profile. A stability rate of 100% indicates that all five runs returned the same label.
- **Format compliance** — proportion of responses conforming to the required JSON schema.
- **Explanation conciseness** — average word count of the generated self-explanation.
- **Output-token consumption** — average number of output tokens / reasoning words per response.
- **Explanation coherence** — consistency between the assigned intent label and the accompanying explanation.

The purpose of the conciseness criteria was to identify a prompt that generated focused explanations suitable for subsequent coding. Shorter explanations were preferred only when they remained coherent and did not reduce classification stability or format compliance.

## Results

All five prompt variants achieved 100% classification stability within each tested profile and 100% JSON-format compliance. The variants nevertheless differed in classification behavior, explanation length, latency, and output-token consumption.

| Prompt | Case ID | Dominant Intent | Stability | Format OK | Avg. Latency (s) | Avg. Reasoning Words |
|---|---:|---|---:|---:|---:|---:|
| V1 | 1 | High Intent | 100% | 100% | 0.9 | 103.8 |
| V1 | 14 | Low Intent | 100% | 100% | 0.6 | 66.8 |
| V1 | 27 | High Intent | 100% | 100% | 0.9 | 103.0 |
| V2 | 1 | Low Intent | 100% | 100% | 0.6 | 45.0 |
| V2 | 14 | Low Intent | 100% | 100% | 0.6 | 55.4 |
| V2 | 27 | High Intent | 100% | 100% | 0.4 | 35.8 |
| V3 | 1 | Low Intent | 100% | 100% | 0.9 | 99.8 |
| V3 | 14 | Low Intent | 100% | 100% | 0.9 | 98.4 |
| V3 | 27 | High Intent | 100% | 100% | 0.8 | 95.2 |
| V4 | 1 | High Intent | 100% | 100% | 1.3 | 110.8 |
| V4 | 14 | Low Intent | 100% | 100% | 0.9 | 84.8 |
| V4 | 27 | High Intent | 100% | 100% | 1.0 | 102.0 |
| V5 | 1 | High Intent | 100% | 100% | 0.5 | 32.8 |
| V5 | 14 | Low Intent | 100% | 100% | 0.6 | 25.2 |
| V5 | 27 | High Intent | 100% | 100% | 1.2 | 58.8 |


## Selected Prompt

`V5` was selected for the main experiment.

Across the three tested profiles, V5 produced an average of approximately 39 explanation words and 66 output tokens per response. It retained complete classification stability and JSON-format compliance while producing substantially shorter explanations than V1, V3, and V4.

V2 also generated relatively concise and fully compliant outputs. V5 was preferred because it provided the best overall combination of concise explanations, low output-token consumption, valid structured responses, and coherent justifications across the tested profiles.

V3 was not selected because it produced comparatively long explanations and, for case ID 1, an explanation that was inconsistent with the assigned classification. This coherence issue was treated as an additional qualitative exclusion criterion.

After selection, V5 was frozen and used unchanged for:

- all synthetic lead profiles;
- all baseline and perturbation conditions;
- all five repeated runs; and
- all three models in the main experiment.

No model-specific prompt optimization was performed.

## Files

The prompt-calibration directory contains the tested prompt variants, calibration scripts, and corresponding result files.

The final prompt used in the main experiment is stored in:

```text
../02_experiment/system_prompt.txt
../02_experiment/user_prompt.txt
```

These files should remain unchanged when reproducing the archived experiment.

## Limitations

- The calibration was conducted using only `meta-llama-3.1-8b-instruct`.
- Only three baseline profiles and five repeated runs per prompt-profile combination were included.
- The calibration assessed technical and output-level suitability rather than predictive validity.
- Stability was evaluated within each prompt-profile combination and does not imply that different prompt variants produced identical classifications.
- Explanation length and output-token consumption are measures of conciseness, not direct measures of explanation quality or faithfulness.
- The selected prompt may not be optimal for every model individually. A single frozen prompt was deliberately used to preserve comparability across models in the main experiment.
- The calibration results reflect the model endpoint and API configuration available at the time of testing.