import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

# Configure API access.
load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

BASE_DIR = Path(__file__).parent

# Limit total concurrent requests across all models.
MAX_WORKERS_TOTAL = 1

# Enforce the global API rate limit with a safety margin.
GLOBAL_MIN_DELAY = 19.0

# Pause all workers after a request error.
ERROR_COOLDOWN = 5.0

# Abort when the server indicates a long quota window.
RETRY_AFTER_ABORT_THRESHOLD = 120  # seconds

# Configure retry backoff.
MAX_ATTEMPTS = 8
BACKOFF_BASE = 2  # seconds (exponential base)
BACKOFF_MAX_JITTER = 4.0  # seconds of uniform jitter added to each wait
SERVER_ERROR_BASE = 6  # flat multiplier for 5xx waits

CHECKPOINT_FILE = BASE_DIR / "results" / "checkpoint.csv"

# Define the models to compare.
models = [
    "meta-llama-3.1-8b-instruct",
    "apertus-70b-instruct-2509",
    "openai-gpt-oss-120b",
]

# Configure file and console logging.
log_dir = BASE_DIR / "logs"
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / "experiment_run.log"

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

# Initialize the OpenAI client.
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=90.0,
    max_retries=0,  # retries are handled manually via call_with_backoff
)

# Load input data and prompt templates.
df = pd.read_csv(
    BASE_DIR.parent / "01_data/synthetic/synthetic_lead_profiles.csv",
    sep=";",
)
df["profile_id"] = df["profile_id"].astype(str)
user_prompt = (BASE_DIR / "user_prompt.txt").read_text(encoding="utf-8")
system_prompt = (BASE_DIR / "system_prompt.txt").read_text(encoding="utf-8")

# Share one rate limiter across all models and worker threads.
_rate_lock = threading.Lock()
_next_ts = 0.0  # monotonic timestamp of the earliest allowed next request

# Signal all workers when a long quota window is reached.
_abort_event = threading.Event()


def _wait_for_slot() -> None:
    """Wait for and reserve the next global request slot."""
    global _next_ts
    with _rate_lock:
        now = time.monotonic()
        wait = _next_ts - now
        if wait > 0:
            time.sleep(wait)
        _next_ts = time.monotonic() + GLOBAL_MIN_DELAY


def _penalise(extra_seconds: float) -> None:
    """Extend the global wait time after an error."""
    global _next_ts
    with _rate_lock:
        target = time.monotonic() + extra_seconds
        if target > _next_ts:
            _next_ts = target


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


def _retry_after_seconds(exc) -> float | None:
    """Extract the Retry-After header in seconds when available."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            value = response.headers.get("retry-after")
            if value is not None:
                return float(value)
        except (ValueError, TypeError, AttributeError):
            pass
    return None


def call_with_backoff(model: str, prompt: str) -> str:
    """Send one completion request with error-specific retry backoff."""
    for attempt in range(MAX_ATTEMPTS):
        # Stop when another worker encounters a long quota window.
        if _abort_event.is_set():
            raise RuntimeError("Run aborted due to long quota wait")

        _wait_for_slot()
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
        except RateLimitError as exc:
            retry_after = _retry_after_seconds(exc)
            if (
                retry_after is not None
                and retry_after > RETRY_AFTER_ABORT_THRESHOLD
            ):
                log.error(
                    f"  [model={model}] Quota window hit – server asks for "
                    f"{retry_after:.0f}s. Aborting run; checkpoint is safe, resume later."
                )
                _abort_event.set()
                raise RuntimeError(
                    f"Long quota wait ({retry_after:.0f}s) – stop & resume later"
                )

            wait = (BACKOFF_BASE**attempt) + random.uniform(
                0,
                BACKOFF_MAX_JITTER,
            )
            # Apply the backoff to every worker.
            _penalise(wait + ERROR_COOLDOWN)
            log.warning(
                f"  [model={model}] 429 Rate Limit – "
                f"wait {wait:.1f}s (+{ERROR_COOLDOWN}s global cooldown) "
                f"(attempt {attempt + 1}/{MAX_ATTEMPTS})"
            )
            time.sleep(wait)
        except APIStatusError as exc:
            if exc.status_code >= 500:
                wait = SERVER_ERROR_BASE * (attempt + 1)
                _penalise(ERROR_COOLDOWN)
                log.warning(
                    f"  [model={model}] {exc.status_code} Server Error – "
                    f"wait {wait}s (attempt {attempt + 1}/{MAX_ATTEMPTS})"
                )
                time.sleep(wait)
            else:
                # Do not retry other 4xx errors.
                raise
        except (APITimeoutError, APIConnectionError) as exc:
            wait = 5 + random.uniform(0, 2)
            _penalise(ERROR_COOLDOWN)
            log.warning(
                f"  [model={model}] {type(exc).__name__} – "
                f"wait {wait:.1f}s (attempt {attempt + 1}/{MAX_ATTEMPTS})"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Max attempts ({MAX_ATTEMPTS}) reached for model: {model}"
    )


def process_job(job):
    """Process one job and return its result."""
    prompt = create_prompt(job["row"])
    log.debug(
        f"  [model={job['model']}] Prompt for profile={job['profile_id']} "
        f"run={job['run']}:\n{prompt}"
    )
    try:
        start = time.time()
        response_text = call_with_backoff(job["model"], prompt)
        latency = time.time() - start
        parsed = parse_response(response_text)
        return {
            "profile_id": job["profile_id"],
            "run": job["run"],
            "model": job["model"],
            "intent": parsed["intent"],
            "reasoning": parsed["reasoning"],
            "parse_error": parsed["parse_error"],
            "latency_s": round(latency, 2),
        }
    except Exception as exc:
        log.error(
            f"  [model={job['model']}] FATAL error on "
            f"profile={job['profile_id']} run={job['run']}: {exc}"
        )
        return {
            "profile_id": job["profile_id"],
            "run": job["run"],
            "model": job["model"],
            "intent": None,
            "reasoning": None,
            "parse_error": str(exc),
            "latency_s": None,
        }


_checkpoint_lock = threading.Lock()


def save_checkpoint(results: list[dict]) -> None:
    """Write the current results to the checkpoint file."""
    CHECKPOINT_FILE.parent.mkdir(exist_ok=True)
    with _checkpoint_lock:
        pd.DataFrame(results).to_csv(
            CHECKPOINT_FILE,
            index=False,
            encoding="utf-8-sig",
        )


def main() -> None:
    # Resume with successful checkpoint rows and retry failed rows.
    if CHECKPOINT_FILE.exists():
        done_df = pd.read_csv(CHECKPOINT_FILE)
        done_df["profile_id"] = done_df["profile_id"].astype(str)
        ok_mask = done_df["intent"].notna()
        ok_df = done_df[ok_mask]
        done_keys = set(
            zip(ok_df["model"], ok_df["profile_id"], ok_df["run"])
        )
        results = ok_df.to_dict("records")
        log.info(
            f"Checkpoint found – {len(done_keys)} successful jobs kept, "
            f"{int((~ok_mask).sum())} failed jobs will be retried."
        )
    else:
        done_keys = set()
        results = []

    # Build jobs for every pending model, run, and profile combination.
    all_jobs = []
    for model in models:
        for run in range(1, 6):
            for _, row in df.iterrows():
                key = (model, row["profile_id"], run)
                if key not in done_keys:
                    all_jobs.append(
                        {
                            "model": model,
                            "run": run,
                            "profile_id": row["profile_id"],
                            "row": row,
                        }
                    )

    # Interleave runs and models in a reproducible order.
    random.seed(42)
    random.shuffle(all_jobs)
    total_pending = len(all_jobs)
    total_overall = total_pending + len(done_keys)
    log.info(
        f"Pending jobs: {total_pending} | Total (incl. checkpoint): {total_overall}"
    )
    log.info(
        f"Concurrency: {MAX_WORKERS_TOTAL} workers (global) | "
        f"min delay: {GLOBAL_MIN_DELAY}s | error cooldown: {ERROR_COOLDOWN}s"
    )
    completed = 0

    # Use one global worker pool for all models.
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_TOTAL) as executor:
            futures = {
                executor.submit(process_job, job): job for job in all_jobs
            }
            for future in as_completed(futures):
                result = future.result()
                job = futures[future]
                with _checkpoint_lock:
                    results.append(result)
                    completed += 1
                    local_done = completed
                log.info(
                    f"[{local_done:>4}/{total_pending}] "
                    f"model={job['model'][:24]:<24} | "
                    f"profile={job['profile_id']:<30} | "
                    f"run={job['run']} | "
                    f"intent={str(result['intent']):<4} | "
                    f"err={'✗ ' + str(result['parse_error'])[:40] if result['parse_error'] else '–'}"
                )
                save_checkpoint(results)
                if _abort_event.is_set():
                    log.warning(
                        "Abort flag set (long quota wait) – stopping ramp-up. "
                        "Checkpoint is safe; re-run later to continue."
                    )
                    break
    except KeyboardInterrupt:
        log.warning("\nKeyboardInterrupt – saving checkpoint before exit …")
        save_checkpoint(results)
        log.info(f"Checkpoint saved ({len(results)} rows). Re-run to continue.")
        return

    # Save the final result table.
    result_path = BASE_DIR / "results" / "llm_lead_intent_results.csv"
    result_path.parent.mkdir(exist_ok=True)
    results_df = (
        pd.DataFrame(results)
        .sort_values(["model", "profile_id", "run"])
        .reset_index(drop=True)
    )
    results_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    # Log the final summary.
    parse_errors = results_df["parse_error"].notna().sum()
    failed_calls = results_df["intent"].isna().sum()
    log.info(f"\n{'─' * 60}")
    log.info(f"Saved: {result_path}")
    log.info(f"Total rows:    {len(results_df)}")
    log.info(f"Failed calls:  {failed_calls}  (intent is null)")
    log.info(f"Parse errors:  {parse_errors}")
    if parse_errors or failed_calls:
        log.warning(
            "  → Some rows failed. Re-run the script to retry only those rows."
        )
    log.info(f"Log:           {log_filename}")
    log.info(f"{'─' * 60}")


if __name__ == "__main__":
    main()
    