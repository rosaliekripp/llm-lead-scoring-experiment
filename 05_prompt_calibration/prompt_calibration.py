import os
from dotenv import load_dotenv
import json
import re
import time
import random
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# Configuration
load_dotenv()
api_key  = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

BASE_DIR = Path(__file__).parent

# Logging setup
log_dir = BASE_DIR / "logs"
log_dir.mkdir(exist_ok=True)
log_filename = log_dir / f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

# Test settings
model         = "meta-llama-3.1-8b-instruct"
n_runs        = 5
VALID_INTENTS = {"High Intent", "Low Intent"}

# Prompt versions to compare (v1 … v5)
PROMPT_VERSIONS = [1, 2, 3, 4, 5]

# CSV rows to test (0-indexed).
# 3 representative profiles from my Maximum Heterogeneity Sample
TEST_CASE_INDICES = [0, 13, 26]

# Initialize client
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=60.0,
    max_retries=0,
)

# Load data
df = pd.read_csv(BASE_DIR.parent / "01_data/synthetic/synthetic_lead_profiles.csv", sep=";")

# Helper functions
def load_prompts(version: int) -> tuple[str, str]:
    """Load system + user prompt for a given version number."""
    sys_path  = BASE_DIR / f"prompts/system_prompt_v{version}.txt"
    user_path = BASE_DIR / f"prompts/user_prompt_v{version}.txt"
    return (
        sys_path.read_text(encoding="utf-8"),
        user_path.read_text(encoding="utf-8"),
    )


def create_prompt(user_template: str, row: pd.Series) -> str:
    """Fill user prompt template with CSV row values."""
    return user_template.format(**row.to_dict())


def count_tokens_approx(text: str) -> int:
    """
    Approximate token count without a tokenizer:
    ~1 token ≈ 4 characters (GPT/Llama rule of thumb).
    Replace with tiktoken or model-specific tokenizer for exact counts.
    """
    return max(1, round(len(text) / 4))


def parse_response(text: str) -> dict:
    """
    Parse JSON model response.
    Returns intent, reasoning, format_ok flag, and parse_error.
    """
    try:
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        data  = json.loads(clean)

        intent    = data.get("intent")
        reasoning = data.get("reasoning")

        # Format conformity: both keys present + intent is a valid label
        format_ok = (
            isinstance(intent, str)
            and isinstance(reasoning, str)
            and intent in VALID_INTENTS
        )

        return {
            "intent":      intent,
            "reasoning":   reasoning,
            "format_ok":   format_ok,
            "parse_error": None,
        }
    except Exception as e:
        return {
            "intent":      None,
            "reasoning":   None,
            "format_ok":   False,
            "parse_error": str(e),
        }


def reasoning_word_count(reasoning: str | None) -> int | None:
    """Count words in the reasoning field."""
    if not reasoning:
        return None
    return len(reasoning.split())


def call_with_backoff(system_prompt: str, user_prompt: str, max_attempts: int = 6) -> dict:
    """
    API call with exponential backoff.
    Returns dict with: response_text, input_tokens_approx, output_tokens_approx.
    """
    for attempt in range(max_attempts):
        try:
            start      = time.time()
            completion = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            latency       = round(time.time() - start, 2)
            response_text = completion.choices[0].message.content

            # Token counts – use API usage object if available, else approximate
            usage = getattr(completion, "usage", None)
            if usage:
                input_tokens  = getattr(usage, "prompt_tokens",     None)
                output_tokens = getattr(usage, "completion_tokens", None)
            else:
                input_tokens  = count_tokens_approx(system_prompt + user_prompt)
                output_tokens = count_tokens_approx(response_text)

            return {
                "response_text":  response_text,
                "latency_s":      latency,
                "input_tokens":   input_tokens,
                "output_tokens":  output_tokens,
            }

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = (2 ** attempt) + random.uniform(0, 2) # max ~66s at attempt 5
                log.info(f"  Rate Limit – wait {wait:.1f}s (Attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)
            elif any(code in err for code in ("500", "502", "503")):
                wait = 5 * (attempt + 1)
                log.info(f"  Server Error – wait {wait}s (Attempt {attempt + 1}/{max_attempts})")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Max attempts ({max_attempts}) reached.")


# Main calibration loop
all_results = []

for version in PROMPT_VERSIONS:
    log.info("\n" + "═" * 60)
    log.info(f"PROMPT VERSION v{version}")
    log.info("═" * 60)

    try:
        system_prompt_tpl, user_prompt_tpl = load_prompts(version)
    except FileNotFoundError as e:
        log.error(f"  Prompt file not found – skipping v{version}: {e}")
        continue

    for case_index in TEST_CASE_INDICES:
        row     = df.iloc[case_index]
        case_id = case_index + 1
        filled_user_prompt = create_prompt(user_prompt_tpl, row)

        log.info(f"\n  ── Case {case_id} (row {case_index}) ──")
        log.info(f"  Profile: {row.to_dict()}")

        run_results = []

        for run in range(1, n_runs + 1):
            log.info(f"\n    Run {run}/{n_runs}")

            try:
                api_result = call_with_backoff(system_prompt_tpl, filled_user_prompt)
                parsed     = parse_response(api_result["response_text"])

                word_count  = reasoning_word_count(parsed["reasoning"])

                log.info(f"    Latency:      {api_result['latency_s']}s")
                log.info(f"    Tokens in/out:{api_result['input_tokens']} / {api_result['output_tokens']}")
                log.info(f"    Raw:          {api_result['response_text']}")
                log.info(f"    Intent:       {parsed['intent']}  |  Format OK: {parsed['format_ok']}")
                log.info(f"    Reasoning words: {word_count}")
                if parsed["parse_error"]:
                    log.warning(f"    ⚠ Parse error: {parsed['parse_error']}")

                run_results.append({
                    "prompt_version":        f"v{version}",
                    "case_id":               case_id,
                    "run":                   run,
                    "model":                 model,
                    "raw_response":          api_result["response_text"],
                    "intent":                parsed["intent"],
                    "reasoning":             parsed["reasoning"],
                    "format_ok":             parsed["format_ok"],
                    "parse_error":           parsed["parse_error"],
                    "reasoning_word_count":  word_count,
                    "latency_s":             api_result["latency_s"],
                    "input_tokens":          api_result["input_tokens"],
                    "output_tokens":         api_result["output_tokens"],
                    "total_tokens":          (
                        (api_result["input_tokens"] or 0) + (api_result["output_tokens"] or 0)
                    ),
                })

            except Exception as e:
                log.error(f"    ✗ API error: {e}")
                run_results.append({
                    "prompt_version":        f"v{version}",
                    "case_id":               case_id,
                    "run":                   run,
                    "model":                 model,
                    "raw_response":          None,
                    "intent":                None,
                    "reasoning":             None,
                    "format_ok":             False,
                    "parse_error":           str(e),
                    "reasoning_word_count":  None,
                    "latency_s":             None,
                    "input_tokens":          None,
                    "output_tokens":         None,
                    "total_tokens":          None,
                })

        # Per-case summary
        run_df = pd.DataFrame(run_results)
        if run_df["intent"].notna().any():
            top_intent     = run_df["intent"].mode()[0]
            stability_rate = (run_df["intent"] == top_intent).mean()
            format_rate    = run_df["format_ok"].mean()
            avg_latency    = run_df["latency_s"].mean()
            avg_tokens_in  = run_df["input_tokens"].mean()
            avg_tokens_out = run_df["output_tokens"].mean()
            avg_words      = run_df["reasoning_word_count"].mean()

            log.info(f"\n  ── Summary v{version} / Case {case_id} ──")
            log.info(f"  Dominant Intent:   '{top_intent}'")
            log.info(f"  Stability Rate:    {stability_rate:.0%}  ({int(stability_rate * n_runs)}/{n_runs})")
            log.info(f"  Format OK Rate:    {format_rate:.0%}")
            log.info(f"  Avg Latency:       {avg_latency:.2f}s")
            log.info(f"  Avg Tokens in/out: {avg_tokens_in:.0f} / {avg_tokens_out:.0f}")
            log.info(f"  Avg Reasoning Wds: {avg_words:.0f}")

        all_results.extend(run_results)

    # Break between versions
    if version != PROMPT_VERSIONS[-1]:
        wait = 5.0
        log.info(f"\n  Pause {wait}s before next prompt version...")
        time.sleep(wait)


# Save raw results
results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

results_df   = pd.DataFrame(all_results)
raw_path     = results_dir / "calibration_raw.csv"
results_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
log.info(f"\n✓ Raw results saved: {raw_path}")

# Aggregated summary table (per version × case)
def stability(series):
    if series.dropna().empty:
        return None
    return (series == series.mode()[0]).mean()

agg = (
    results_df
    .groupby(["prompt_version", "case_id"])
    .agg(
        dominant_intent   = ("intent",                  lambda s: s.mode()[0] if s.notna().any() else None),
        stability_rate    = ("intent",                  stability),
        format_ok_rate    = ("format_ok",               "mean"),
        avg_latency_s     = ("latency_s",               "mean"),
        avg_input_tokens  = ("input_tokens",            "mean"),
        avg_output_tokens = ("output_tokens",           "mean"),
        avg_total_tokens  = ("total_tokens",            "mean"),
        avg_reasoning_wds = ("reasoning_word_count",    "mean"),
        parse_errors      = ("parse_error",             lambda s: s.notna().sum()),
    )
    .reset_index()
)

# Round for readability
for col in ["stability_rate", "format_ok_rate"]:
    agg[col] = agg[col].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "–")
for col in ["avg_latency_s", "avg_input_tokens", "avg_output_tokens",
            "avg_total_tokens", "avg_reasoning_wds"]:
    agg[col] = agg[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "–")

summary_path = results_dir / "calibration_summary.csv"
agg.to_csv(summary_path, index=False, encoding="utf-8-sig")

log.info("\n" + "═" * 60)
log.info("CALIBRATION SUMMARY")
log.info("═" * 60)
log.info("\n" + agg.to_string(index=False))
log.info(f"\n✓ Summary saved: {summary_path}")
log.info(f"✓ Log saved:     {log_filename}")