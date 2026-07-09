import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent 
csv_path = BASE.parent.parent / "02_experiment" / "results" / "llm_lead_intent_results.csv"

# Load data
df = pd.read_csv(csv_path, encoding="utf-8-sig")

# Drop failed parses
df = df[df["parse_error"].isna()].copy()

# Standardize intent: 1 -> “High Intent”, 2 -> “Low Intent”
df["intent"] = df["intent"].replace({
    "1": "High Intent", "2": "Low Intent",   # if String
    1: "High Intent", 2: "Low Intent"        # if Integer
})

# Remove the “parse_error” and “latency_s” columns
df = df.drop(columns=["parse_error", "latency_s"])

# Save cleaned data
df.to_csv(BASE / "llm_lead_intent_results_clean.csv", index=False, encoding="utf-8-sig")