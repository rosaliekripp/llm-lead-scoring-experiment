import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Configure API access.
load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

BASE_DIR = Path(__file__).parent

# Configure file and console logging.
log_dir = BASE_DIR / "logs"
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8", mode="a"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger()

# Configure the test run.
model = "meta-llama-3.1-8b-instruct"
case_index = 0
n_runs = 5

# Initialize the OpenAI client.
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=60.0,
    max_retries=0,  # retries are handled manually via call_with_backoff
)

# Load input data and prompt templates.
df = pd.read_csv(
    Path("01_data/synthetic/synthetic_lead_profiles.csv"),
    sep=";",
)
user_prompt = (BASE_DIR / "user_prompt.txt").read_text(encoding="utf-8")
system_prompt = (BASE_DIR / "system_prompt.txt").read_text(encoding="utf-8")


def create_prompt(row):
    """Create the user prompt from a CSV row."""
    return user_prompt.format(**row.to_dict())


def parse_response(text):
    """Parse a JSON response after removing Markdown fences."""
    try:
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        data = json.loads(clean)
        return {
            "intent": data.get("intent"),
            "reasoning": data.get("reasoning"),
            "parse_error": None,
        }
    except Exception as exc:
        return {
            "intent": None,
            "reasoning": None,
            "parse_error": str(exc),
        }


def call_with_backoff(model, prompt, max_attempts=4):
    """Call the API with backoff for rate-limit and server errors."""
    for attempt in range(max_attempts):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return completion.choices[0].message.content

        except Exception as exc:
            error = str(exc)
            if "429" in error or "rate" in error.lower():
                wait = (2**attempt) + random.uniform(0, 1)
                log.info(
                    f"  Rate Limit – wait {wait:.1f}s "
                    f"(Attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(wait)
            elif any(code in error for code in ("500", "502", "503")):
                wait = 5 * (attempt + 1)
                log.info(
                    f"  Server Error – wait {wait}s "
                    f"(Attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(
        f"Max attempts ({max_attempts}) reached for model: {model}"
    )


# Select the test case.
row = df.iloc[case_index]
case_id = case_index + 1
prompt = create_prompt(row)

log.info("=" * 60)
log.info("Test Case:")
log.info(row.to_dict())
log.info("\nUser Prompt:")
log.info(prompt)
log.info("=" * 60)

# Run the selected case repeatedly.
results = []
latencies = []

for run in range(1, n_runs + 1):
    log.info(f"\n--- Run {run}/{n_runs} ---")

    try:
        start = time.time()
        response_text = call_with_backoff(model, prompt)
        latency = round(time.time() - start, 2)
        latencies.append(latency)

        parsed = parse_response(response_text)

        log.info(f"Latency:    {latency}s")
        log.info(f"Raw:       {response_text}")
        log.info(f"Parsed:    {parsed}")

        results.append(
            {
                "case_id": case_id,
                "run": run,
                "model": model,
                "prompt": prompt,
                "raw_response": response_text,
                "intent": parsed["intent"],
                "reasoning": parsed["reasoning"],
                "parse_error": parsed["parse_error"],
                "latency_s": latency,
            }
        )

    except Exception as exc:
        log.info(f"Error: {exc}")

        results.append(
            {
                "case_id": case_id,
                "run": run,
                "model": model,
                "prompt": prompt,
                "raw_response": None,
                "intent": None,
                "reasoning": None,
                "parse_error": str(exc),
                "latency_s": None,
            }
        )

# Append the results to the CSV file.
results_df = pd.DataFrame(results)
result_path = BASE_DIR / "results" / "llm_lead_intent_test_results.csv"
write_header = not result_path.exists()
results_df.to_csv(
    result_path,
    mode="a",
    index=False,
    header=write_header,
    encoding="utf-8-sig",
)

# Log the result summary.
log.info("\n" + "=" * 60)
log.info("CONCLUSION")
log.info("=" * 60)

log.info("\nResults:")
log.info(
    results_df[["run", "intent", "parse_error", "latency_s"]].to_string(
        index=False
    )
)

log.info("\nIntent Distribution:")
log.info(results_df["intent"].value_counts(dropna=False))

# Calculate the share of the most common response.
if results_df["intent"].notna().any():
    top_intent = results_df["intent"].mode()[0]
    stability_rate = (results_df["intent"] == top_intent).sum() / n_runs
    log.info(
        f"\nStability Rate: {stability_rate:.0%} "
        f"({int(stability_rate * n_runs)}/{n_runs} Runs → '{top_intent}')"
    )

# Summarize successful request latencies.
if latencies:
    log.info(
        f"\nLatency – Ø {sum(latencies) / len(latencies):.2f}s | "
        f"Min {min(latencies):.2f}s | Max {max(latencies):.2f}s"
    )

parse_errors = results_df["parse_error"].notna().sum()
if parse_errors:
    log.info(f"\n⚠ Parse Error: {parse_errors}/{n_runs}")

print("\nSaved: llm_lead_intent_test_results.csv")
log.info(f"\nRun finished. Log saved in: {log_filename}")