import csv
import logging
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Configure the screening test.
load_dotenv()

MODEL = "meta-llama-3.1-8b-instruct"
CONCURRENCY_LEVELS = [1, 2, 4]
CALLS_PER_LEVEL = 10
TOTAL_CALLS_TARGET = 975
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

# Configure results file and console logging.
BASE_DIR = Path(__file__).parent
results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)
results_filename = results_dir / "api_screening_test_results.csv"
log_dir = BASE_DIR / "logs"
log_dir.mkdir(exist_ok=True)
log_filename = (
    log_dir
    / f"api_screening_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
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
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    timeout=60.0,
    max_retries=0,
)


def single_call(call_id: int) -> dict:
    """Run one API call with retries for selected server errors."""
    for attempt in range(3):
        start = time.time()
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {"role": "user", "content": SAMPLE_PROMPT},
                ],
            )
            latency = round(time.time() - start, 3)
            return {
                "call_id": call_id,
                "latency": latency,
                "status": "ok",
                "response": completion.choices[0].message.content.strip(),
                "error": None,
            }
        except Exception as exc:
            error = str(exc)
            if (
                any(code in error for code in ("500", "502", "503"))
                and attempt < 2
            ):
                log.info(
                    f"  ↻ Call {call_id} – 500 error, "
                    f"retry {attempt + 1}/2 ..."
                )
                time.sleep(3)
                continue
            latency = round(time.time() - start, 3)
            status = (
                "rate_limit"
                if "429" in error
                else (
                    "server_error"
                    if any(
                        code in error for code in ("500", "502", "503")
                    )
                    else "error"
                )
            )
            return {
                "call_id": call_id,
                "latency": latency,
                "status": status,
                "response": None,
                "error": error,
            }


def run_stage(concurrency: int, n_calls: int) -> list[dict]:
    """Run one screening stage at the selected concurrency level."""
    log.info(f"\n{'─' * 50}")
    log.info(f"Stage: concurrency={concurrency}, calls={n_calls}")
    log.info(f"{'─' * 50}")

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(single_call, call_id): call_id
            for call_id in range(1, n_calls + 1)
        }
        for future in as_completed(futures):
            result = future.result()
            icon = "✓" if result["status"] == "ok" else "✗"
            log.info(
                f"  {icon} Call {result['call_id']:>3} | "
                f"{result['latency']}s | {result['status']} | "
                f"{result['response'] or result['error'][:60]}"
            )
            results.append(result)
            if result["status"] != "ok":
                time.sleep(5)
    return results


def save_summary_csv(all_results: list[dict]) -> None:
    """Append the summary metrics to the CSV results file."""
    file_exists = results_filename.exists() and results_filename.stat().st_size > 0
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with results_filename.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "timestamp",
            "model",
            "concurrency",
            "calls",
            "ok",
            "errors",
            "avg_seconds",
            "p95_seconds",
            "max_seconds",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header only when creating a new CSV file.
        if not file_exists:
            writer.writeheader()

        # Calculate and append metrics for each concurrency stage.
        for stage_results in all_results:
            concurrency = stage_results["concurrency"]
            results = stage_results["results"]
            successful_results = [
                result for result in results if result["status"] == "ok"
            ]
            errors = len(results) - len(successful_results)
            latencies = (
                sorted(
                    result["latency"]
                    for result in successful_results
                )
                if successful_results
                else [0]
            )
            avg_latency = round(statistics.mean(latencies), 3)
            p95 = (
                round(latencies[int(len(latencies) * 0.95) - 1], 3)
                if len(latencies) > 1
                else latencies[0]
            )
            max_latency = max(latencies)

            writer.writerow(
                {
                    "timestamp": run_timestamp,
                    "model": MODEL,
                    "concurrency": concurrency,
                    "calls": len(results),
                    "ok": len(successful_results),
                    "errors": errors,
                    "avg_seconds": avg_latency,
                    "p95_seconds": p95,
                    "max_seconds": max_latency,
                }
            )

    log.info(f"CSV results saved: {results_filename}")


def print_summary(all_results: list[dict]) -> None:
    """Log a summary for all completed concurrency stages."""
    log.info(f"\n{'=' * 50}")
    log.info("SCREENING TEST SUMMARY")
    log.info(f"{'=' * 50}")
    log.info(
        f"{'Concurrency':<14} {'Calls':>6} {'OK':>6} {'Errors':>8} "
        f"{'Avg(s)':>8} {'p95(s)':>8} {'Max(s)':>8}"
    )
    log.info("─" * 62)

    # Calculate metrics for each concurrency stage.
    for stage_results in all_results:
        concurrency = stage_results["concurrency"]
        results = stage_results["results"]
        successful_results = [
            result for result in results if result["status"] == "ok"
        ]
        errors = len(results) - len(successful_results)
        latencies = (
            sorted(result["latency"] for result in successful_results)
            if successful_results
            else [0]
        )
        avg_latency = round(statistics.mean(latencies), 3)
        p95 = (
            round(latencies[int(len(latencies) * 0.95) - 1], 3)
            if len(latencies) > 1
            else latencies[0]
        )
        max_latency = max(latencies)
        log.info(
            f"{concurrency:<14} {len(results):>6} "
            f"{len(successful_results):>6} {errors:>8} "
            f"{avg_latency:>8} {p95:>8} {max_latency:>8}"
        )

    log.info(f"\nLog saved: {log_filename}")


def main() -> None:
    """Run each screening stage and print the final summary."""
    log.info("=" * 50)
    log.info(f"API Screening Test – Model: {MODEL}")
    log.info(f"Planned full run: {TOTAL_CALLS_TARGET} calls")
    log.info("=" * 50)

    all_results = []
    for concurrency in CONCURRENCY_LEVELS:
        results = run_stage(
            concurrency=concurrency,
            n_calls=CALLS_PER_LEVEL,
        )
        ok_rate = (
            sum(1 for result in results if result["status"] == "ok")
            / len(results)
        )
        all_results.append(
            {"concurrency": concurrency, "results": results}
        )

        log.info(f"  → OK rate: {ok_rate:.0%}")
        if ok_rate < 0.5:
            log.info("  ⚠ OK rate below 50% – stopping ramp-up.")
            break

        # Pause briefly between stages.
        time.sleep(5)

    print_summary(all_results)
    save_summary_csv(all_results)


if __name__ == "__main__":
    main()
