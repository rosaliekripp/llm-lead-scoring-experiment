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
log_filename = (
    log_dir / f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

# Configure the calibration run.
model = "meta-llama-3.1-8b-instruct"
n_runs = 5
VALID_INTENTS = {"High Intent", "Low Intent"}

# Define the prompt versions to compare.
PROMPT_VERSIONS = [1, 2, 3, 4, 5]

# Select representative profiles by zero-based row index.
TEST_CASE_INDICES = [0, 13, 26]

# Initialize the OpenAI client.
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=60.0,
    max_retries=0,
)

# Load the input data.
df = pd.read_csv(
    BASE_DIR.parent / "01_data/synthetic/synthetic_lead_profiles.csv",
    sep=";",
)


def load_prompts(version: int) -> tuple[str, str]:
    """Load the system and user prompts for one version."""
    sys_path = BASE_DIR / f"prompts/system_prompt_v{version}.txt"
    user_path = BASE_DIR / f"prompts/user_prompt_v{version}.txt"
    return (
        sys_path.read_text(encoding="utf-8"),
        user_path.read_text(encoding="utf-8"),
    )


def create_prompt(user_template: str, row: pd.Series) -> str:
    """Create the user prompt from a template and CSV row."""
    return user_template.format(**row.to_dict())


def count_tokens_approx(text: str) -> int:
    """Approximate the token count using four characters per token."""
    return max(1, round(len(text) / 4))


def parse_response(text: str) -> dict:
    """Parse the model response and validate its format."""
    try:
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        data = json.loads(clean)

        intent = data.get("intent")
        reasoning = data.get("reasoning")

        # Require both fields and a supported intent label.
        format_ok = (
            isinstance(intent, str)
            and isinstance(reasoning, str)
            and intent in VALID_INTENTS
        )

        return {
            "intent": intent,
            "reasoning": reasoning,
            "format_ok": format_ok,
            "parse_error": None,
        }
    except Exception as exc:
        return {
            "intent": None,
            "reasoning": None,
            "format_ok": False,
            "parse_error": str(exc),
        }


def reasoning_word_count(reasoning: str | None) -> int | None:
    """Count words in the reasoning field."""
    if not reasoning:
        return None
    return len(reasoning.split())


def call_with_backoff(
    system_prompt: str,
    user_prompt: str,
    max_attempts: int = 6,
) -> dict:
    """Call the API with exponential backoff and return response metrics."""
    for attempt in range(max_attempts):
        try:
            start = time.time()
            completion = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            latency = round(time.time() - start, 2)
            response_text = completion.choices[0].message.content

            # Use API token counts when available, otherwise estimate them.
            usage = getattr(completion, "usage", None)
            if usage:
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)
            else:
                input_tokens = count_tokens_approx(system_prompt + user_prompt)
                output_tokens = count_tokens_approx(response_text)

            return {
                "response_text": response_text,
                "latency_s": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        except Exception as exc:
            error = str(exc)
            if "429" in error or "rate" in error.lower():
                wait = (2**attempt) + random.uniform(0, 2)
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

    raise RuntimeError(f"Max attempts ({max_attempts}) reached.")


# Run the calibration for each prompt version.
all_results = []

for version in PROMPT_VERSIONS:
    log.info("\n" + "═" * 60)
    log.info(f"PROMPT VERSION v{version}")
    log.info("═" * 60)

    try:
        system_prompt_tpl, user_prompt_tpl = load_prompts(version)
    except FileNotFoundError as exc:
        log.error(f"  Prompt file not found – skipping v{version}: {exc}")
        continue

    for case_index in TEST_CASE_INDICES:
        row = df.iloc[case_index]
        case_id = case_index + 1
        filled_user_prompt = create_prompt(user_prompt_tpl, row)

        log.info(f"\n  ── Case {case_id} (row {case_index}) ──")
        log.info(f"  Profile: {row.to_dict()}")

        run_results = []

        for run in range(1, n_runs + 1):
            log.info(f"\n    Run {run}/{n_runs}")

            try:
                api_result = call_with_backoff(
                    system_prompt_tpl,
                    filled_user_prompt,
                )
                parsed = parse_response(api_result["response_text"])

                word_count = reasoning_word_count(parsed["reasoning"])

                log.info(f"    Latency:      {api_result['latency_s']}s")
                log.info(
                    f"    Tokens in/out:{api_result['input_tokens']} / "
                    f"{api_result['output_tokens']}"
                )
                log.info(f"    Raw:          {api_result['response_text']}")
                log.info(
                    f"    Intent:       {parsed['intent']}  |  "
                    f"Format OK: {parsed['format_ok']}"
                )
                log.info(f"    Reasoning words: {word_count}")
                if parsed["parse_error"]:
                    log.warning(f"    ⚠ Parse error: {parsed['parse_error']}")

                run_results.append(
                    {
                        "prompt_version": f"v{version}",
                        "case_id": case_id,
                        "run": run,
                        "model": model,
                        "raw_response": api_result["response_text"],
                        "intent": parsed["intent"],
                        "reasoning": parsed["reasoning"],
                        "format_ok": parsed["format_ok"],
                        "parse_error": parsed["parse_error"],
                        "reasoning_word_count": word_count,
                        "latency_s": api_result["latency_s"],
                        "input_tokens": api_result["input_tokens"],
                        "output_tokens": api_result["output_tokens"],
                        "total_tokens": (
                            (api_result["input_tokens"] or 0)
                            + (api_result["output_tokens"] or 0)
                        ),
                    }
                )

            except Exception as exc:
                log.error(f"    ✗ API error: {exc}")
                run_results.append(
                    {
                        "prompt_version": f"v{version}",
                        "case_id": case_id,
                        "run": run,
                        "model": model,
                        "raw_response": None,
                        "intent": None,
                        "reasoning": None,
                        "format_ok": False,
                        "parse_error": str(exc),
                        "reasoning_word_count": None,
                        "latency_s": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                    }
                )

        # Summarize each test case.
        run_df = pd.DataFrame(run_results)
        if run_df["intent"].notna().any():
            top_intent = run_df["intent"].mode()[0]
            stability_rate = (run_df["intent"] == top_intent).mean()
            format_rate = run_df["format_ok"].mean()
            avg_latency = run_df["latency_s"].mean()
            avg_tokens_in = run_df["input_tokens"].mean()
            avg_tokens_out = run_df["output_tokens"].mean()
            avg_words = run_df["reasoning_word_count"].mean()

            log.info(f"\n  ── Summary v{version} / Case {case_id} ──")
            log.info(f"  Dominant Intent:   '{top_intent}'")
            log.info(
                f"  Stability Rate:    {stability_rate:.0%}  "
                f"({int(stability_rate * n_runs)}/{n_runs})"
            )
            log.info(f"  Format OK Rate:    {format_rate:.0%}")
            log.info(f"  Avg Latency:       {avg_latency:.2f}s")
            log.info(
                f"  Avg Tokens in/out: {avg_tokens_in:.0f} / "
                f"{avg_tokens_out:.0f}"
            )
            log.info(f"  Avg Reasoning Wds: {avg_words:.0f}")

        all_results.extend(run_results)

    # Pause between prompt versions.
    if version != PROMPT_VERSIONS[-1]:
        wait = 5.0
        log.info(f"\n  Pause {wait}s before next prompt version...")
        time.sleep(wait)


# Save the raw calibration results.
results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

results_df = pd.DataFrame(all_results)
raw_path = results_dir / "calibration_raw.csv"
results_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
log.info(f"\n✓ Raw results saved: {raw_path}")


# Calculate stability for one grouped result series.
def stability(series):
    """Return the share of the most common non-null value."""
    if series.dropna().empty:
        return None
    return (series == series.mode()[0]).mean()


# Aggregate results by prompt version and test case.
agg = (
    results_df.groupby(["prompt_version", "case_id"])
    .agg(
        dominant_intent=(
            "intent",
            lambda s: s.mode()[0] if s.notna().any() else None,
        ),
        stability_rate=("intent", stability),
        format_ok_rate=("format_ok", "mean"),
        avg_latency_s=("latency_s", "mean"),
        avg_input_tokens=("input_tokens", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
        avg_total_tokens=("total_tokens", "mean"),
        avg_reasoning_wds=("reasoning_word_count", "mean"),
        parse_errors=("parse_error", lambda s: s.notna().sum()),
    )
    .reset_index()
)

# Format summary values for readability.
for col in ["stability_rate", "format_ok_rate"]:
    agg[col] = agg[col].apply(
        lambda x: f"{x:.0%}" if pd.notna(x) else "–"
    )

for col in [
    "avg_latency_s",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_total_tokens",
    "avg_reasoning_wds",
]:
    agg[col] = agg[col].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "–"
    )

summary_path = results_dir / "calibration_summary.csv"
agg.to_csv(summary_path, index=False, encoding="utf-8-sig")

log.info("\n" + "═" * 60)
log.info("CALIBRATION SUMMARY")
log.info("═" * 60)
log.info("\n" + agg.to_string(index=False))
log.info(f"\n✓ Summary saved: {summary_path}")
log.info(f"✓ Log saved:     {log_filename}")
