import os
from dotenv import load_dotenv
import json
import re
import time
import random
import threading
import pandas as pd
import logging
from pathlib import Path
from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError, APIConnectionError
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure API access
load_dotenv()
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

BASE_DIR = Path(__file__).parent

# Total parallel in-flight requests across ALL models combined.
# The academic API enforces its rate limit per API key (globally), NOT per model,
# so we cap the TOTAL concurrency. At ~200 req/hour a single worker is plenty.
MAX_WORKERS_TOTAL = 1

# Global token-bucket floor: minimum pause between ANY two requests, across all
# models and all workers combined (seconds). Sized for the 200/hour limit
# (3600 / 200 = 18s) plus 1s safety margin.
GLOBAL_MIN_DELAY = 19.0

# After ANY error (429/5xx/timeout), additionally pause the global bucket by this
# much, so all workers back off together instead of hammering the API again.
ERROR_COOLDOWN = 5.0

# If the server's Retry-After exceeds this, a longer quota window (hour/day) is
# exhausted: abort the whole run cleanly instead of blocking for minutes.
RETRY_AFTER_ABORT_THRESHOLD = 120  # seconds

# Backoff settings
MAX_ATTEMPTS        = 8
BACKOFF_BASE        = 2          # seconds (exponential base)
BACKOFF_MAX_JITTER  = 4.0        # seconds of uniform jitter added to each wait
SERVER_ERROR_BASE   = 6          # flat multiplier for 5xx waits

CHECKPOINT_FILE = BASE_DIR / "results" / "checkpoint.csv"

# Define models for comparison
models = [
    "meta-llama-3.1-8b-instruct",
    "apertus-70b-instruct-2509",
    "openai-gpt-oss-120b",
]

# Logging setup
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

# Initialize OpenAI client
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=90.0,
    max_retries=0,   # retries are handled manually via call_with_backoff
)

# Load input data and prompt templates
df            = pd.read_csv(BASE_DIR.parent / "01_data/synthetic/synthetic_lead_profiles.csv", sep=";")
df["profile_id"] = df["profile_id"].astype(str)
user_prompt   = (BASE_DIR / "user_prompt.txt").read_text(encoding="utf-8")
system_prompt = (BASE_DIR / "system_prompt.txt").read_text(encoding="utf-8")

# Global rate limiter (shared across all models and all worker threads)

_rate_lock = threading.Lock()
_next_ts   = 0.0   # monotonic timestamp of the earliest allowed next request

# Set when a long quota window is hit, so every worker stops at the next loop.
_abort_event = threading.Event()


def _wait_for_slot() -> None:
    """Block until the global inter-request delay has elapsed, then reserve
    the next slot. One single bucket for the whole process."""
    global _next_ts
    with _rate_lock:
        now  = time.monotonic()
        wait = _next_ts - now
        if wait > 0:
            time.sleep(wait)
        _next_ts = time.monotonic() + GLOBAL_MIN_DELAY


def _penalise(extra_seconds: float) -> None:
    """Push the global 'next allowed request' timestamp further into the future
    so that ALL workers back off together after an error."""
    global _next_ts
    with _rate_lock:
        target = time.monotonic() + extra_seconds
        if target > _next_ts:
            _next_ts = target


# Define helper functions
def create_prompt(row):
    """Creates the user prompt from a CSV row."""
    return user_prompt.format(**row.to_dict())


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


def _retry_after_seconds(exc) -> float | None:
    """Extract the Retry-After header (seconds) from an exception, if present."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            val = resp.headers.get("retry-after")
            if val is not None:
                return float(val)
        except (ValueError, TypeError, AttributeError):
            pass
    return None


def call_with_backoff(model: str, prompt: str) -> str:
    """
    Send one completion request with exponential backoff.
    Handles:
      - 429 RateLimitError  → honour Retry-After if present, else exponential
                              backoff + jitter, plus a GLOBAL cooldown so every
                              worker pauses together. A very long Retry-After
                              (> RETRY_AFTER_ABORT_THRESHOLD) aborts the whole run.
      - 5xx APIStatusError  → linear backoff (server-side issue)
      - APITimeoutError     → short flat wait, then retry
      - APIConnectionError  → short flat wait, then retry
    Unknown errors are re-raised immediately.
    """
    for attempt in range(MAX_ATTEMPTS):
        # If another worker hit a long quota window, stop immediately.
        if _abort_event.is_set():
            raise RuntimeError("Run aborted due to long quota wait")

        _wait_for_slot()
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
        except RateLimitError as e:
            retry_after = _retry_after_seconds(e)
            if retry_after is not None and retry_after > RETRY_AFTER_ABORT_THRESHOLD:
                log.error(
                    f"  [model={model}] Quota window hit – server asks for "
                    f"{retry_after:.0f}s. Aborting run; checkpoint is safe, resume later."
                )
                _abort_event.set()
                raise RuntimeError(f"Long quota wait ({retry_after:.0f}s) – stop & resume later")
            else:
                wait = (BACKOFF_BASE ** attempt) + random.uniform(0, BACKOFF_MAX_JITTER)
            # Make ALL workers back off, not just this one.
            _penalise(wait + ERROR_COOLDOWN)
            log.warning(
                f"  [model={model}] 429 Rate Limit – "
                f"wait {wait:.1f}s (+{ERROR_COOLDOWN}s global cooldown) "
                f"(attempt {attempt + 1}/{MAX_ATTEMPTS})"
            )
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code >= 500:
                wait = SERVER_ERROR_BASE * (attempt + 1)
                _penalise(ERROR_COOLDOWN)
                log.warning(
                    f"  [model={model}] {e.status_code} Server Error – "
                    f"wait {wait}s (attempt {attempt + 1}/{MAX_ATTEMPTS})"
                )
                time.sleep(wait)
            else:
                # 4xx errors other than 429 are not retriable
                raise
        except (APITimeoutError, APIConnectionError) as e:
            wait = 5 + random.uniform(0, 2)
            _penalise(ERROR_COOLDOWN)
            log.warning(
                f"  [model={model}] {type(e).__name__} – "
                f"wait {wait:.1f}s (attempt {attempt + 1}/{MAX_ATTEMPTS})"
            )
            time.sleep(wait)
    raise RuntimeError(
        f"Max attempts ({MAX_ATTEMPTS}) reached for model: {model}"
    )


def process_job(job):
    """Processes a single job and returns the result dictionary."""
    prompt = create_prompt(job["row"])
    log.debug(f"  [model={job['model']}] Prompt for profile={job['profile_id']} run={job['run']}:\n{prompt}")
    try:
        start         = time.time()
        response_text = call_with_backoff(job["model"], prompt)
        latency       = time.time() - start
        parsed        = parse_response(response_text)
        return {
            "profile_id":   job["profile_id"],
            "run":          job["run"],
            "model":        job["model"],
            "intent":       parsed["intent"],
            "reasoning":    parsed["reasoning"],
            "parse_error":  parsed["parse_error"],
            "latency_s":    round(latency, 2),
        }
    except Exception as e:
        log.error(
            f"  [model={job['model']}] FATAL error on "
            f"profile={job['profile_id']} run={job['run']}: {e}"
        )
        return {
            "profile_id":   job["profile_id"],
            "run":          job["run"],
            "model":        job["model"],
            "intent":       None,
            "reasoning":    None,
            "parse_error":  str(e),
            "latency_s":    None,
        }


# Load checkpoint
_checkpoint_lock = threading.Lock()

def save_checkpoint(results: list[dict]) -> None:
    CHECKPOINT_FILE.parent.mkdir(exist_ok=True)
    with _checkpoint_lock:
        pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False, encoding="utf-8-sig")


def main() -> None:
    # Resume from checkpoint — but ONLY count successful rows as "done".
    # Rows that failed previously (intent is null / parse_error set) are retried.
    if CHECKPOINT_FILE.exists():
        done_df = pd.read_csv(CHECKPOINT_FILE)
        done_df["profile_id"] = done_df["profile_id"].astype(str)
        ok_mask   = done_df["intent"].notna()
        ok_df     = done_df[ok_mask]
        done_keys = set(zip(ok_df["model"], ok_df["profile_id"], ok_df["run"]))
        # Keep only successful rows in the working set; failed ones get re-run.
        results = ok_df.to_dict("records")
        log.info(
            f"Checkpoint found – {len(done_keys)} successful jobs kept, "
            f"{int((~ok_mask).sum())} failed jobs will be retried."
        )
    else:
        done_keys = set()
        results   = []
    # Build job list (one entry per model × run × profile not yet done)
    all_jobs = []
    for model in models:                    # 3 models
        for run in range(1, 6):             # 5 runs
            for _, row in df.iterrows():    # profiles
                key = (model, row["profile_id"], run)
                if key not in done_keys:
                    all_jobs.append({
                        "model":      model,
                        "run":        run,
                        "profile_id": row["profile_id"],
                        "row":        row,
                    })
    # Shuffle so runs/models are interleaved (avoids hammering one model in a row)
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
    # One single global pool over ALL jobs/models guarantees that there are never
    # more than MAX_WORKERS_TOTAL requests in flight against the shared API key.
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_TOTAL) as executor:
            futures = {executor.submit(process_job, job): job for job in all_jobs}
            for future in as_completed(futures):
                result = future.result()
                job    = futures[future]
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
    # Save final results
    result_path = BASE_DIR / "results" / "llm_lead_intent_results.csv"
    result_path.parent.mkdir(exist_ok=True)
    results_df  = (
        pd.DataFrame(results)
        .sort_values(["model", "profile_id", "run"])
        .reset_index(drop=True)
    )
    results_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    # Summary
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