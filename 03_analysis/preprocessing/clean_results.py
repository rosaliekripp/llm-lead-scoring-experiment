"""Clean and standardize the raw lead-intent experiment results.

The script removes observations with parsing errors, standardizes the intent
labels, drops processing-related columns, and saves the cleaned dataset for
subsequent analyses.
"""

from pathlib import Path

import pandas as pd

# Configure the input and output paths relative to this script.
BASE_DIR = Path(__file__).resolve().parent
csv_path = (
    BASE_DIR.parent.parent
    / "02_experiment"
    / "results"
    / "llm_lead_intent_results.csv"
)

# Load the raw experiment results.
df = pd.read_csv(csv_path, encoding="utf-8-sig")

# Retain only successfully parsed model responses.
df = df[df["parse_error"].isna()].copy()

# Standardize string and integer intent codes to the final text labels.
df["intent"] = df["intent"].replace(
    {
        "1": "High Intent",
        "2": "Low Intent",
        1: "High Intent",
        2: "Low Intent",
    }
)

# Remove processing-related columns that are not required for analysis.
df = df.drop(columns=["parse_error", "latency_s"])

# Save the cleaned data for the subsequent hypothesis analyses.
df.to_csv(
    BASE_DIR / "llm_lead_intent_results_clean.csv",
    index=False,
    encoding="utf-8-sig",
)
