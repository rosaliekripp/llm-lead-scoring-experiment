import re
from pathlib import Path
import pandas as pd

# paths
base_dir = Path(__file__).parent

# load data
df = pd.read_csv(base_dir / "llm_lead_intent_results_clean.csv")

# features that could be perturbed, matches the profile_id naming
FEATURES = ["company_size", "website_dwell_time", "demo_requests",
            "email_response", "industry", "region"]

# surface terms that may refer to a feature inside the explanation text
FEATURE_TERMS = {
    "company_size": ["company size", "employees", "size (", "10",
                     "80", "250", "500", "1200",
                     "5000", "10000", "large potential",
                     "small company", "large company", "company of"],
    "website_dwell_time": ["dwell time", "website dwell", "min)", "minutes",
                           "sec)", "seconds", "time on site", "time spent",
                           "15 sec", "40 sec", "1 min", "2.5 min", "4 min",
                           "7 min", "10 min"],
    "demo_requests": ["demo request", "demo requests", "demo", "no demo",
                      "multiple demo", "requested a demo"],
    "email_response": ["email response", "email", "response", "reply",
                       "message", "wrote", "looks interesting",
                       "send more info", "not interested", "no interest"],
    "industry": ["industry", "finance", "public service", "software", "IT",
                 "manufacturing", "healthcare", "energy", "retail", "sector"],
    "region": ["region", "europe", "european", "north america", "south america",
               "americas", "africa", "african", "middle east", "asia", "asian",
               "location", "geograph", "based in"],
}

# marker sets for the coding scheme
NEUTRAL_MARKERS = [" with ", "along with", "as well as"]
REFUTING_MARKERS = ["despite", "although", "even though", "though", "however",
                    "but ", "while", "regardless", "nevertheless"]
SUPPORTING_MARKERS = ["indicate", "indicates", "suggest", "suggests", "because",
                      "drives", "driven", "support this", "supports", "reflect",
                      "shows strong", "signal", "signals", "further support"]

# derive the perturbed feature from the profile_id
def perturbed_feature(profile_id):
    # base rows have no perturbed feature
    if profile_id.endswith("_BASE") or profile_id.split("_", 1)[-1] == "BASE":
        return "BASE"
    rest = profile_id.split("_", 1)[1] if "_" in profile_id else ""
    for f in FEATURES:
        if rest.startswith(f):
            return f
    return "unknown"

# find the sentence that mentions the feature, else the whole text
def relevant_span(text, feature):
    terms = FEATURE_TERMS.get(feature, [])
    sentences = re.split(r"(?<=[.!?])\s+", str(text))
    for s in sentences:
        low = s.lower()
        if any(t in low for t in terms):
            return s
    return ""

# helper: does the span contain other features than the target one
def mentions_other_feature(scope, target_feature):
    # check surface terms of all features except the target
    for f, terms in FEATURE_TERMS.items():
        if f == target_feature:
            continue
        if any(t in scope for t in terms):
            return True
    return False

# find the position of the first matching term of a feature in a text
def first_pos(scope, terms):
    positions = [scope.find(t) for t in terms if t in scope]
    return min(positions) if positions else -1

# split a scope into the part before and after the first contrastive marker
def split_on_contrast(scope):
    positions = [(scope.find(m), m) for m in REFUTING_MARKERS if m in scope]
    positions = [(p, m) for p, m in positions if p != -1]
    if not positions:
        return scope, ""
    pos, mark = min(positions)
    return scope[:pos], scope[pos + len(mark):]

# suggest a code 0 to 3 for the feature in the given text
def suggest_code(text, feature):
    if feature in ("BASE", "unknown"):
        return "", ""
    low = str(text).lower()
    terms = FEATURE_TERMS.get(feature, [])
    mentioned = any(t in low for t in terms)
    if not mentioned:
        return 0, "not mentioned"
    span = relevant_span(text, feature).lower()
    scope = span if span else low

    # handle contrastive sentences by looking at which side the feature is on
    if any(m in scope for m in REFUTING_MARKERS):
        before, after = split_on_contrast(scope)
        feat_before = any(t in before for t in terms)
        feat_after = any(t in after for t in terms)
        # feature on the refuted side, before the contrast marker
        if feat_before and not feat_after:
            return 2, "refuted side of contrast"
        # feature on the supporting side, after the contrast marker
        if feat_after and any(m in after for m in SUPPORTING_MARKERS):
            return 3, "supporting side of contrast"
        if feat_after:
            return 1, "mentioned after contrast no marker"
        # feature appears on both sides, needs manual review
        return 2, "feature on both sides of contrast"

    # supporting marker present, but check if causality belongs to the group
    if any(m in scope for m in SUPPORTING_MARKERS):
        feat_pos = first_pos(scope, terms)
        neutral_pos = first_pos(scope, [m.strip() for m in NEUTRAL_MARKERS])
        if (feat_pos != -1 and neutral_pos != -1 and feat_pos < neutral_pos
                and mentions_other_feature(scope[neutral_pos:], feature)):
            return 1, "enumeration causality on other features"
        return 3, "supporting marker"

    if any(m in scope for m in NEUTRAL_MARKERS):
        return 1, "neutral mention"
    return 1, "mention no marker"

# build the coding template
rows = []
for _, r in df.iterrows():
    feat = perturbed_feature(r["profile_id"])
    code, reason = suggest_code(r["reasoning"], feat)
    rows.append({
        "profile_id": r["profile_id"],
        "run": r["run"],
        "model": r["model"],
        "perturbed_feature": feat,
        "intent": r["intent"],
        "reasoning": r["reasoning"],
        "suggested_code": code,
        "suggestion_reason": reason,
        "final_code": code,   # you overwrite this column during review
        "checked": "",        # mark with x once reviewed
    })

coding = pd.DataFrame(rows)

# drop base rows since there is no perturbed feature to code
coding_to_review = coding[~coding["perturbed_feature"].isin(["BASE", "unknown"])].copy()

# save both the full file and the review file
coding_to_review.to_csv(base_dir / "explanation_coding_template.csv", index=False)

# short summary
print("rows total:", len(coding))
print("rows to review:", len(coding_to_review))
print("\nsuggested code distribution:")
print(coding_to_review["suggested_code"].value_counts(dropna=False).sort_index())
print("\nby feature:")
print(coding_to_review.groupby("perturbed_feature")["suggested_code"]
      .value_counts().unstack(fill_value=0))