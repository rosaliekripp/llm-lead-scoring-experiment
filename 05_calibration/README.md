# Calibration

## Purpose

Before running the main experiment, a prompt calibration was conducted to identify the most suitable prompt version for the lead scoring task. The goal was to select a prompt that produces stable, correctly formatted, and concise outputs across representative lead profiles — not to compare model behaviour, but to fix a single prompt for use across all three models in the main experiment.

## Tested Variants

Five prompt versions (v1–v5) were evaluated against three representative lead profiles drawn from the Maximum Heterogeneity Sample (case IDs 1, 14, 27), covering distinct combinations of engagement level and company size. Each version was run 5 times per profile (n = 75 total calls) using `meta-llama-3.1-8b-instruct`.

## Selection Criteria

Versions were assessed on:

- **Stability rate** — consistency of the intent label across 5 runs (target: 100%)
- **Format OK rate** — JSON output conformity with the required schema (target: 100%)
- **Reasoning conciseness** — average word count of the reasoning field; shorter outputs that focus on decision-relevant features are preferred over exhaustive feature enumeration
- **Token efficiency** — lower output token counts indicate focused feature selection rather than redundant elaboration

All five versions achieved 100% stability and format conformity. **v5 was selected** based on its superior token efficiency (avg. ~66 output tokens) and most concise reasoning (avg. ~39 words), indicating selective feature attribution rather than mechanical enumeration of all input variables. Longer reasoning outputs (v1, v3, v4) did not provide additional analytical value for the explainability evaluation. v3 additionally produced internally inconsistent reasoning for case 1 (A_BASE), which further disqualified it.

## Results

| prompt_version | case_id | dominant_intent | stability_rate | format_ok_rate | avg_latency_s | avg_reasoning_wds |
|---|---|---|---|---|---|---|
| v1 | 1 | High Intent | 100% | 100% | 0.9 | 103.8 |
| v1 | 14 | Low Intent | 100% | 100% | 0.6 | 66.8 |
| v1 | 27 | High Intent | 100% | 100% | 0.9 | 103.0 |
| v2 | 1 | Low Intent | 100% | 100% | 0.6 | 45.0 |
| v2 | 14 | Low Intent | 100% | 100% | 0.6 | 55.4 |
| v2 | 27 | High Intent | 100% | 100% | 0.4 | 35.8 |
| v3 | 1 | Low Intent | 100% | 100% | 0.9 | 99.8 |
| v3 | 14 | Low Intent | 100% | 100% | 0.9 | 98.4 |
| v3 | 27 | High Intent | 100% | 100% | 0.8 | 95.2 |
| v4 | 1 | High Intent | 100% | 100% | 1.3 | 110.8 |
| v4 | 14 | Low Intent | 100% | 100% | 0.9 | 84.8 |
| v4 | 27 | High Intent | 100% | 100% | 1.0 | 102.0 |
| v5 | 1 | High Intent | 100% | 100% | 0.5 | 32.8 |
| v5 | 14 | Low Intent | 100% | 100% | 0.6 | 25.2 |
| v5 | 27 | High Intent | 100% | 100% | 1.2 | 58.8 |

## Limitations of the Prompt Selection

Only tested using `meta-llama-3.1-8b-instruct`.