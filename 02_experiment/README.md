# Experiment

This directory contains the prompts, execution scripts, logs, and model outputs for the LLM lead scoring experiment.

## Data Generation Process

The experiment loads the five synthetic reference profiles from `../01_data/synthetic/synthetic_lead_profiles.csv`.

For each profile, it generates:

- one unchanged baseline condition; and
- twelve one-factor-at-a-time perturbations, created by setting each of the six lead features to a predefined low or high value.

This produces 65 input conditions:

```text
5 profiles × [1 baseline + (6 features × 2 levels)] = 65 conditions
```

Each condition is submitted to three models in five repeated runs, resulting in 975 planned model outputs. The execution order is randomized across profiles, conditions, runs, and models.

## Prompt Setup

The experiment uses two prompt files:

- `system_prompt.txt` — defines the vendor context, scoring task, and required JSON output;
- `user_prompt.txt` — provides the structured lead profile for each model call.

The selected prompt was calibrated before the main experiment and then frozen. It is used unchanged across all models and conditions. No explicit scoring rules, feature weights, or decision thresholds are provided.

Each model returns:

- a binary classification: `High intent` or `Low intent`; and
- a short free-text explanation.

## LLM Configuration

The main experiment compares:

- Meta Llama 3.1 8B Instruct;
- Apertus 70B Instruct; and
- OpenAI GPT-OSS 120B.

All models are accessed through the GWDG Chat AI API and use the same configuration. Temperature is set to `0`; other decoding parameters retain their provider defaults.

The experiment is executed serially with a single worker. Failed requests are retried using exponential backoff, and successful responses are stored through checkpointing.

API credentials must be supplied in a local `.env` file or through environment variables. Credentials are not included in the repository and must never be committed to Git.

## Running the Experiment

To test one case:

```bash
python3 02_experiment/run_single_case.py
```

To run the complete experiment:

```bash
python3 02_experiment/run_full_experiment.py
```

The full experiment may generate API costs and is subject to the model provider's access restrictions and rate limits.

## Test Run and API Stress Test

The preliminary API stress test evaluated model availability, latency, error rates, concurrency behavior, and structured-output reliability. It was used only for technical model screening and was not part of the main experiment.

The stress-test scripts and results are located in:

```text
../04_model_selection/
```

## Outputs

Experiment outputs are stored in `results/`:

- `checkpoint.csv` — checkpoint data used to resume interrupted execution;
- `llm_lead_intent_results.csv` — archived outputs from the main experiment;
- `llm_lead_intent_test_results.csv` — outputs from preliminary test runs.

Runtime information and error messages may be stored in `logs/`. Existing archived results should not be overwritten when reproducing the reported analyses.