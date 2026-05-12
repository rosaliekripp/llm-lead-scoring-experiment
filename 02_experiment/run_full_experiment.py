import os
from dotenv import load_dotenv
import json
import re
import time
import random
import pandas as pd
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure API access
load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

CHECKPOINT_FILE = "checkpoint.csv"
MAX_WORKERS     = 5 

# Define models for comparison
models = [
    "mistral-large-3-675b-instruct-2512",
    "qwen3.5-122b-a10b",
    "llama-3.3-70b-instruct",
]

# Initialize OpenAI client
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=60.0,
    max_retries=0,   # Retries are controlled manually using `call_with_backoff`
)

# Load input data and prompt templates
df              = pd.read_csv("example_dataset.csv", sep=";")
prompt_template = Path("prompt_template.txt").read_text(encoding="utf-8")
system_prompt   = Path("system_prompt.txt").read_text(encoding="utf-8")

# Define helper functions
def create_prompt(row):
    """Creates the user prompt from a CSV row."""
    return prompt_template.format(**row.to_dict())


def parse_response(text):
    """Parses the model response as JSON; removes Markdown fences beforehand."""
    try:
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        data  = json.loads(clean)
        return {
            "intent":      data.get("intent"),
            "reasoning":   data.get("reasoning"),
            "parse_error": None,
        }
    except Exception as e:
        return {
            "intent":      None,
            "reasoning":   None,
            "parse_error": str(e),
        }


def call_with_backoff(model, prompt, max_attempts=4):
    """API call with exponential backoff for rate-limit and server errors."""
    for attempt in range(max_attempts):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt},
                ],
            )
            return completion.choices[0].message.content

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"  Rate limit – wait {wait:.1f}s (Attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)
            elif any(code in err for code in ("500", "502", "503")):
                wait = 5 * (attempt + 1)
                print(f"  Server Error – wait {wait}s (Attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)
            else:
                raise  # Report unknown errors immediately

    raise RuntimeError(f"Max attempts ({max_attempts}) reached for model: {model}")


def process_job(job):
    """Processes a single job and returns the result dictionary."""
    prompt = create_prompt(job["row"])
    try:
        start         = time.time()
        response_text = call_with_backoff(job["model"], prompt)
        latency       = time.time() - start
        print(f"  Latency: {latency:.2f}s (model={job['model']})")
        parsed        = parse_response(response_text)
        return {
            "case_id":      job["case_id"],
            "run":          job["run"],
            "model":        job["model"],
            "raw_response": response_text,
            "intent":       parsed["intent"],
            "reasoning":    parsed["reasoning"],
            "parse_error":  parsed["parse_error"],
            "latency_s":    round(latency, 2),
        }
    except Exception as e:
        return {
            "case_id":      job["case_id"],
            "run":          job["run"],
            "model":        job["model"],
            "raw_response": None,
            "intent":       None,
            "reasoning":    None,
            "parse_error":  str(e),
            "latency_s":    None,
        }


# Load Checkpoint
if Path(CHECKPOINT_FILE).exists():
    done_df   = pd.read_csv(CHECKPOINT_FILE)
    done_keys = set(zip(done_df["model"], done_df["case_id"], done_df["run"]))
    results   = done_df.to_dict("records")
    print(f"Checkpoint found – {len(done_keys)} Jobs have already been completed, will continue.")
else:
    done_keys = set()
    results   = []

# Build randomized experiment jobs
jobs = []
for model in models:
    for run in range(1, 6):
        for idx, row in df.iterrows():
            job = {
                "model":   model,
                "run":     run,
                "case_id": idx + 1,
                "row":     row,
            }
            key = (model, idx + 1, run)
            if key not in done_keys:   # Skip jobs that have already been completed
                jobs.append(job)

jobs_df      = pd.DataFrame(jobs).sample(frac=1, random_state=42).reset_index(drop=True)
pending_jobs = jobs_df.to_dict("records")   # ThreadPoolExecutor requires serializable dictionaries

print(f"Pending Jobs: {len(pending_jobs)} | In total: {len(jobs_df) + len(done_keys)}")

# Run experiment and collect responses
completed = 0
total     = len(pending_jobs)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_job, job): job for job in pending_jobs}

    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        completed += 1

        job = futures[future]
        print(
            f"[{completed}/{total}] "
            f"model={job['model']} | case={job['case_id']} | run={job['run']} | "
            f"intent={result['intent']} | error={result['parse_error']}"
        )

        # Refresh the checkpoint after each completed job
        pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")

# Save results to CSV
results_df = pd.DataFrame(results).sort_values(["model", "case_id", "run"]).reset_index(drop=True)
results_df.to_csv("llm_lead_intent_results.csv", index=False, encoding="utf-8-sig")

# Preview saved results
print(results_df.head())
print("Gespeichert: llm_lead_intent_results.csv")