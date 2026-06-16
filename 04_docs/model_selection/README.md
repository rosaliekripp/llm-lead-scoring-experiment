# Model Selection

## Purpose

Before finalising the experimental setup, an API stress test was conducted to identify which of the available open-source models on the GWDG API (Gesellschaft für wissenschaftliche Datenverarbeitung mbH Göttingen) were suitable for the main experiment. The goal was not to evaluate model quality, but to verify that each candidate model could sustain the full experimental load of N = 975 structured inference calls reliably and within acceptable latency bounds.

## Candidate Models

All models with a primary text generation / instruction-following capability available via the GWDG API at the time of testing were included as candidates:

- `apertus-70b-instruct-2509`
- `gemma-4-31b-it`
- `glm-4.7`
- `meta-llama-3.1-8b-instruct`
- `mistral-large-3-675b-instruct-2512` (tested across three independent runs)
- `openai-gpt-oss-120b`
- `qwen3.6-35b-a3b`

## Stress Test Design

Each model was tested across three concurrency levels (1, 2, and 4 parallel workers) with 10 calls per level (30 calls total per model). Each call used the same structured lead scoring prompt with JSON output format. Recorded metrics: success rate, average latency, p95 latency, and maximum latency per stage.

## Exclusion Criteria

A model was excluded if it met any of the following conditions:

- **Error rate > 30%** across all concurrency levels combined
- **Test abort** triggered by sub-50% success rate at any single stage
- **Average latency consistently > 5 s**, posing a risk of systematic timeout errors at N = 975

## Results

| Model | OK / Calls | Error Rate | Avg (s) | Max (s) | Decision |
|---|---|---|---|---|---|
| apertus-70b-instruct-2509 | 30/30 | 0% | 2.20 | 2.64 | ✓ Selected |
| gemma-4-31b-it | 29/30 | 3% | 1.32 | 4.59 | Excluded¹ |
| glm-4.7 | 28/30 | 7% | 3.91 | 38.23 | Excluded² |
| meta-llama-3.1-8b-instruct | 30/30 | 0% | 1.25 | 2.60 | ✓ Selected |
| mistral-large-3-675b (run 1, June 15 a.m.) | 11/20 | 45% | 2.22 | 3.34 | — |
| mistral-large-3-675b (run 2, June 15 p.m.) | 16/30 | 47% | 2.59 | 3.18 | — |
| mistral-large-3-675b (run 3, June 16 a.m.) | 30/30 | 0% | 5.25 | 24.47 | Excluded³ |
| openai-gpt-oss-120b | 30/30 | 0% | 0.94 | 1.22 | ✓ Selected |
| qwen3.6-35b-a3b | 29/30 | 3% | 7.48 | 10.44 | Excluded⁴ |

¹ Technically eligible, but excluded in favour of higher architectural heterogeneity across the selected set.  
² Extreme latency outlier (38.2 s at concurrency 1); unpredictable single-threaded performance.  
³ Zero errors in run 3, but maximum latencies of 17.3 s and 24.5 s at concurrency levels 1 and 4 indicate unacceptable latency variability for a 975-call experiment. Runs 1-2 additionally confirmed error instability on June 15.  
⁴ Average latency 7-8 s across all stages; p95 exceeding 10 s under moderate concurrency.

## Selected Models

| Model | Parameters | Origin | Role in Experiment |
|---|---|---|---|
| `meta-llama-3.1-8b-instruct` | 8B | Meta AI (USA) | Efficiency baseline |
| `apertus-70b-instruct-2509` | 70B | Swiss AI Initiative / EPFL, ETH Zürich, CSCS (Europe) | High-parameter, open-science reference |
| `openai-gpt-oss-120b` | 120B | OpenAI (USA) | Large-scale performance anchor |

The three models span distinct points in the design space with respect to parameter count (8B / 70B / 120B), training philosophy, and organisational origin, providing the architectural heterogeneity required for the cross-model comparisons specified in H4.