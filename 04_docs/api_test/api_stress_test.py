"""
API Stress Test
Measures throughput, latency, and error behaviour under load.
Ramp up concurrency until errors appear or a target rate is reached.
"""

import os
import time
import statistics
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

# Configuration
load_dotenv()

MODEL           = "meta-llama-3.1-8b-instruct"
CONCURRENCY_LEVELS  = [1, 2, 4]       # parallel workers per stage
CALLS_PER_LEVEL     = 10                      # calls per concurrency level
TOTAL_CALLS_TARGET  = 960                     # informational – not enforced here
SAMPLE_PROMPT = """The company is a European B2B IT consulting firm offering data-driven mid-market platforms for customer and sales analytics with a focus on integration and AI-supported decision-making.

Task:
Classify the lead as High Intent or Low Intent and explain your reasoning by mentioning the specific variable.

Return only valid JSON in this format:
{
    "intent": "High Intent or Low Intent",
    "reasoning": "short explanation mentioning the specific variables"
}

Lead data:
- Company size: 250 employees
- Industry: Financial Services
- Region: Europe
- Website time on site: 8 minutes
- Demo requests: 2
- Email response: Yes"""

# Logging setup
BASE_DIR = Path(__file__).parent
log_filename = BASE_DIR / f"api_stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

# Client
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    timeout=60.0,
    max_retries=0,
)

# Single call
def single_call(call_id: int) -> dict:
    for attempt in range(3):   # max 3 Versuche
        start = time.time()
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": SAMPLE_PROMPT},
                ],
            )
            latency = round(time.time() - start, 3)
            return {
                "call_id":  call_id,
                "latency":  latency,
                "status":   "ok",
                "response": completion.choices[0].message.content.strip(),
                "error":    None,
            }
        except Exception as e:
            err = str(e)
            if any(code in err for code in ("500", "502", "503")) and attempt < 2:
                log.info(f"  ↻ Call {call_id} – 500 error, retry {attempt + 1}/2 ...")
                time.sleep(3)
                continue
            latency = round(time.time() - start, 3)
            status = "rate_limit" if "429" in err else "server_error" if any(
                c in err for c in ("500", "502", "503")) else "error"
            return {
                "call_id":  call_id,
                "latency":  latency,
                "status":   status,
                "response": None,
                "error":    err,
            }

# Stage runner
def run_stage(concurrency: int, n_calls: int) -> list[dict]:
    log.info(f"\n{'─'*50}")
    log.info(f"Stage: concurrency={concurrency}, calls={n_calls}")
    log.info(f"{'─'*50}")

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(single_call, i): i for i in range(1, n_calls + 1)}
        for future in as_completed(futures):
            r = future.result()
            icon = "✓" if r["status"] == "ok" else "✗"
            log.info(f"  {icon} Call {r['call_id']:>3} | {r['latency']}s | "
                    f"{r['status']} | {r['response'] or r['error'][:60]}")
            results.append(r)
            if r["status"] != "ok":
                time.sleep(5)   # short break after each error
    return results

# Summary
def print_summary(all_results: list[dict]):
    log.info(f"\n{'='*50}")
    log.info("STRESS TEST SUMMARY")
    log.info(f"{'='*50}")
    log.info(f"{'Concurrency':<14} {'Calls':>6} {'OK':>6} {'Errors':>8} "
             f"{'Avg(s)':>8} {'p95(s)':>8} {'Max(s)':>8}")
    log.info("─" * 62)

    # Group by concurrency level (stored in result dict via stage)
    from itertools import groupby
    for stage_results in all_results:
        conc    = stage_results["concurrency"]
        res     = stage_results["results"]
        ok      = [r for r in res if r["status"] == "ok"]
        errs    = len(res) - len(ok)
        lats    = sorted(r["latency"] for r in ok) if ok else [0]
        avg_lat = round(statistics.mean(lats), 3)
        p95     = round(lats[int(len(lats) * 0.95) - 1], 3) if len(lats) > 1 else lats[0]
        max_lat = max(lats)
        log.info(f"{conc:<14} {len(res):>6} {len(ok):>6} {errs:>8} "
                 f"{avg_lat:>8} {p95:>8} {max_lat:>8}")

    log.info(f"\nLog saved: {log_filename}")

# Main
def main():
    log.info("=" * 50)
    log.info(f"API Stress Test – Model: {MODEL}")
    log.info(f"Planned full run: {TOTAL_CALLS_TARGET} calls")
    log.info("=" * 50)

    all_results = []
    for conc in CONCURRENCY_LEVELS:
        results = run_stage(concurrency=conc, n_calls=CALLS_PER_LEVEL)
        ok_rate = sum(1 for r in results if r["status"] == "ok") / len(results)
        all_results.append({"concurrency": conc, "results": results})

        log.info(f"  → OK rate: {ok_rate:.0%}")
        if ok_rate < 0.5:
            log.info("  ⚠ OK rate below 50% – stopping ramp-up.")
            break

        # Brief cooldown between stages
        time.sleep(5)

    print_summary(all_results)


if __name__ == "__main__":
    main()