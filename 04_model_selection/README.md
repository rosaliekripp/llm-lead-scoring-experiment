# Model Selection

## Purpose

Before finalizing the experimental setup, an API screening test was conducted to determine which candidate models available through the GWDG Chat AI API were technically suitable for the main experiment.

The screening did not evaluate model quality or lead scoring performance. Its purpose was to assess whether each model could support the planned 975 structured inference calls with sufficiently stable availability and reliable JSON output.

The GWDG API imposed per-key rate limits of 30 requests per minute, 200 requests per hour, and 1,000 requests per day at the time of data collection. The main experiment was therefore executed serially with a single worker and a fixed delay between requests. Consequently, throughput and performance under parallel load were not primary selection criteria. The screening focused mainly on availability, response reliability, structured-output compliance, and latency stability.

## Candidate Models

All available models with a primary text-generation or instruction-following capability that met the initial access requirements were considered:

- `apertus-70b-instruct-2509`
- `gemma-4-31b-it`
- `glm-4.7`
- `meta-llama-3.1-8b-instruct`
- `mistral-large-3-675b-instruct-2512`
- `openai-gpt-oss-120b`
- `qwen3.6-35b-a3b`

The screening was restricted to models available through the GWDG API at the time of testing. It does not represent an exhaustive comparison of all available LLMs.

## Screening Test Design

Each model was tested at three concurrency levels:
- 1 parallel worker;
- 2 parallel workers; and
- 4 parallel workers.

Each stage included 10 planned calls, resulting in up to 30 calls per screening run. All calls used the same structured lead scoring prompt and required a JSON response.

The following metrics were recorded:
- number of successful and failed calls;
- success and error rates;
- JSON-output compliance;
- average latency;
- 95th-percentile latency; and
- maximum latency.

Latency was recorded as a diagnostic metric. Because the main experiment was executed serially, absolute latency under parallel load was considered secondary to availability and response stability.
A screening run could be terminated early if the success rate at a concurrency stage fell below 50%. As a result, some models received fewer than the planned 30 calls.

## Technical Eligibility Criteria
A model was classified as technically unsuitable if at least one of the following conditions applied:
- an overall error rate above 30%;
- early termination caused by a success rate below 50% at any stage;
- repeated or session-dependent availability failures;
- unreliable structured JSON output; or
- severe latency instability, such as extreme outliers associated with timeout risk.
High but stable latency alone was not treated as an automatic exclusion criterion. Technical eligibility was assessed separately from the final design-based selection of three models.

## Results

The reported average latency is the unweighted mean of the three stage-level averages for concurrency levels 1, 2, and 4. Because each completed stage contained the same number of planned calls, this corresponds to an equal weighting of the concurrency stages. The reported maximum is the highest observed single-call latency across all completed stages.
The error rate is the proportion of failed calls among all calls issued to the model. For prematurely terminated runs, the denominator reflects the number of calls completed before the stopping rule was applied.

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
⁴ Persistent multi-second response times (avg 7-8 s, p95 > 10 s). Technically eligible, but not selected because of comparatively high latency and the deliberately limited three-model design.

## Selected Models

| Model | Parameter Scale | Organizational Origin | Role in the Experiment |
|---|---:|---|---|
| `meta-llama-3.1-8b-instruct` | 8B | Meta AI, USA | Smaller open-weight reference model |
| `apertus-70b-instruct-2509` | 70B | Swiss AI Initiative, Europe | European open-science reference model |
| `openai-gpt-oss-120b` | 120B | OpenAI, USA | Larger open-weight reference model |

The final model set was selected to cover distinct points in the design space with respect to parameter scale, development context, and organizational origin. This deliberate heterogeneity supports the cross-model comparisons specified under H4.
The selection is not intended to represent all available LLMs. The screening established technical suitability, whereas the final choice additionally reflected the study design and the need to keep the experiment feasible within the applicable API rate limits.

## Relationship to the Main Experiment

The concurrency screening was conducted only for technical model selection. It was not part of the 975-call main experiment, and its outputs were not included in the hypothesis analyses.

The main experiment used:
- one worker;
- serial request execution;
- a fixed delay between calls;
- automatic retries with exponential backoff; and
- checkpointing to support recovery after interruptions.

## Limitations

- The screening reflects model availability and API behavior only at the time of testing.
- Provider infrastructure and model endpoints may change over time.
- The small number of calls per concurrency level limits the precision of latency and error-rate estimates.
- Concurrency results should not be interpreted as a comprehensive throughput benchmark.
- Model quality was not evaluated during screening.
- The final selection was intentionally restricted to three models and should not be generalized to all LLMs.