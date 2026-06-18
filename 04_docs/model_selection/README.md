# Model Selection

## Purpose

Before finalising the experimental setup, an API screening test was conducted to identify which of the available open-source models on the GWDG API (Gesellschaft für wissenschaftliche Datenverarbeitung mbH Göttingen) were suitable for the main experiment. The goal was not to evaluate model quality, but to verify that each candidate model could **reliably serve N = 975 structured inference calls** with stable availability and consistently valid JSON output.
Because the GWDG API enforces a per-key rate limit (30 requests/minute, 200/hour, 1,000/day), the main experiment was executed **serially** with a single worker and a fixed inter-request delay. Absolute latency and behaviour under parallel load were therefore *not* decisive for the experiment. The screening test consequently served primarily to exclude models with **unstable availability or unreliable structured output**, rather than to benchmark throughput.

## Candidate Models

All models with a primary text generation / instruction-following capability available via the GWDG API at the time of testing were included as candidates:

- `apertus-70b-instruct-2509`
- `gemma-4-31b-it`
- `glm-4.7`
- `meta-llama-3.1-8b-instruct`
- `mistral-large-3-675b-instruct-2512` (tested across three independent runs)
- `openai-gpt-oss-120b`
- `qwen3.6-35b-a3b`

## Screening Test Design

Each model was tested across three concurrency levels (1, 2, and 4 parallel workers) with 10 calls per level (30 calls total per model). Each call used the same structured lead scoring prompt with JSON output format. **Recorded metrics:** success rate, response stability, and latency (average, p95, maximum). Latency was recorded for completeness but, given the serial execution of the main experiment,was treated as a secondary indicator only.

## Exclusion Criteria

All models have met the JSON output format requirement.
A model was excluded if it met any of the following conditions:

- **Error rate > 30%** across all stages combined (unreliable availability)
- **Test abort** triggered by a sub-50% success rate at any single stage
- **Severe latency instability** (e.g. multi-second p95/max outliers) indicating
  unpredictable response behaviour that could cause sporadic timeout errors even
  under serial execution

## Results

In the summary table, the reported average latency is the unweighted mean of the three per-stage average latencies (concurrency 1, 2, 4; n = 10 each), and the reported maximum is the highest single-call latency observed across all stages. The error rate is the share of failed calls among all calls issued to the model.

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

¹ Technically eligible (low error rate, stable output). **Not excluded on technical
grounds** but deliberately omitted as a design decision: the final set was chosen to
span distinct parameter classes and organisational origins (see *Selected Models*),
and gemma overlapped with already-covered design points. Its exclusion is a
selection choice, not a reliability finding.  
² Extreme latency outlier (38.2 s at concurrency 1); unpredictable single-threaded
response behaviour, posing a sporadic timeout risk even under serial execution.  
³ Reliability could not be reproduced across sessions: runs 1-2 (June 15) showed
45% and 47% error rates, i.e. the model failed to respond reliably on half of the
test sessions. Although run 3 was error-free, the documented session-to-session
instability disqualifies it for a reproducible N = 975 experiment.  
⁴ Persistent multi-second response times (avg 7-8 s, p95 > 10 s) indicating unstable
response behaviour rather than a load artefact.

## Selected Models

| Model | Parameters | Origin | Role in Experiment |
|---|---|---|---|
| `meta-llama-3.1-8b-instruct` | 8B | Meta AI (USA) | Efficiency baseline |
| `apertus-70b-instruct-2509` | 70B | Swiss AI Initiative / EPFL, ETH Zürich, CSCS (Europe) | High-parameter, open-science reference |
| `openai-gpt-oss-120b` | 120B | OpenAI (USA) | Large-scale performance anchor |

The three models were selected to span **distinct points in the design space** with respect to parameter count (8B / 70B / 120B), training philosophy, and organisational origin (Meta, OpenAI, and the European open-science Apertus initiative). This deliberate heterogeneity — rather than an exhaustive enumeration of all available models — provides the architectural contrast required for the cross-model comparisons specified in H4. The screening test ensured that every selected model meets the reliability and output-validity requirements of the serial N = 975 setup.

## Limitations of the Model Selection

The selection is intentionally restricted to three models positioned along the axes of **parameter size** and **organisational origin**. Other technically eligible models (e.g. `gemma-4-31b-it`) were not included, in order to keep the comparison interpretable and the data-collection effort within the API rate limits. The screening test prioritised availability and structured-output reliability over absolute latency, since the main experiment was executed serially; conclusions about model behaviour under high parallel load are therefore outside its scope.