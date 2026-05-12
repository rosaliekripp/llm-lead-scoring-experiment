import pandas as pd

results_df = pd.read_csv("llm_lead_intent_results.csv")
#results_df = pd.read_csv("llm_lead_intent_test_results.csv")

stability = (
    results_df
    .groupby(["model", "case_id"])["intent"]
    .nunique()
    .reset_index(name="n_unique_intents")
)

stability["stable"] = stability["n_unique_intents"] == 1

print(stability)

model_stability = (
    stability
    .groupby("model")["stable"]
    .mean()
    .reset_index(name="share_stable_cases")
)

print(model_stability)