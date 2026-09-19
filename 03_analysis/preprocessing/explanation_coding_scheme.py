import re
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

# Load data
df = pd.read_csv(BASE_DIR / "llm_lead_intent_results_clean.csv")

# Features that could be perturbed, matches the profile_id naming
FEATURES = ["company_size", "industry", "region", "website_dwell_time",
            "demo_requests", "email_response"]

# Surface terms that may refer to a feature inside the explanation text.
FEATURE_TERMS = {
    "company_size": ["company_size", "company size", "employees",
                     "large potential", "small company", "large company",
                     "company of", "organization", "organisation", "enterprise",
                     "company is", "company ("],
    "industry": ["industry", "finance", "public service", "software", "healthcare",
                 "energy", "retail", "manufacturing", "sector", "software / it"],
    "region": ["region", "europe", "european", "north america", "south america",
               "americas", "africa", "african", "middle east", "asia", "asian",
               "location", "geograph", "based in"],
    "website_dwell_time": ["website_dwell_time", "dwell time", "website dwell",
                           "time on site", "time spent", "dwell", "website ("],
    "demo_requests": ["demo_requests", "demo request", "demo requests", "demos",
                      "demo", "no demo", "multiple demo", "requested a demo"],
    "email_response": ["email_response", "email response", "email", "e-mail"],
}

# Concessive markers: the clause AFTER the marker is the refuted side,
# so a feature BEFORE the marker sits on the carrying/supporting side.
CONCESSIVE_MARKERS = ["despite", "although", "even though", "though",
                      "regardless", "nevertheless"]
# Adversative markers: the clause BEFORE the marker is the refuted side,
# the clause AFTER carries the outcome.
ADVERSATIVE_MARKERS = ["but ", ", but", "however", "whereas"]
# Verb-internal contrast: "X outweighs/overrides Y" -> X supports, Y refuted.
OUTWEIGH_MARKERS = ["outweigh", "outweighs", "outweighed", "outweighing",
                    "override", "overrides", "overriding"]

NEUTRAL_MARKERS = [" with ", "along with", "as well as", "neutral"]

# Finite causal verbs
SUPPORTING_MARKERS = ["indicate", "indicates", "suggest", "suggests", "because",
                      "drives", "driven", "support this", "supports", "reflect",
                      "reflects", "show strong", "shows strong", "signal", "signals",
                      "further support", "match our target", "matches our target",
                      "strengthen", "strengthens", "reinforce", "reinforced",
                      "reinforces", "typical target", "strong indicator",
                      "key indicator", "key focus"]
# Trailing participles that span a preceding enumeration and attribute an effect
# to every listed feature.
SPANNING_PARTICIPLES = ["indicating", "suggesting", "reflecting", "signaling",
                        "signalling", "reinforcing", "driving"]

# Nouns whose "with" is a complement ("engagement with", "interest with") and
# therefore NOT an accompaniment/enumeration marker.
WITH_COMPLEMENT_NOUNS = ["engagement", "interest", "satisfaction", "experience",
                         "familiarity", "involvement", "interaction", "dwell"]

POS_WORDS = ["strong", "high", "active", "interest", "potential", "engag", "promis",
             "readiness", "positive", "relevant", "fit", "match", "above average",
             "significant", "openness", "growth", "key"]
NEG_WORDS = ["low", "lack", " no ", "not ", "limited", "short", "neutral",
             "disinterest", "weak", "small", "less", "without", "absence"]

# Feature detection
def perturbed_feature(profile_id):
    if profile_id.endswith("_BASE") or profile_id.split("_", 1)[-1] == "BASE":
        return "BASE"
    rest = profile_id.split("_", 1)[1] if "_" in profile_id else ""
    for f in FEATURES:
        if rest.startswith(f):
            return f
    return "unknown"

def _terms(feature):
    return FEATURE_TERMS.get(feature, [])

def mentions(scope, feature):
    return any(t in scope for t in _terms(feature))

def first_pos(scope, feature):
    positions = [scope.find(t) for t in _terms(feature) if t in scope]
    return min(positions) if positions else -1

def mentions_other_feature(scope, target_feature):
    for f, terms in FEATURE_TERMS.items():
        if f == target_feature:
            continue
        if any(t in scope for t in terms):
            return True
    return False

# Sentence handling (quote-safe)
def split_sentences(text):
    """Split into sentences without breaking inside quotes."""
    text = str(text)
    quotes = []

    def _stash(m):
        quotes.append(m.group(0))
        return "\x00%d\x00" % (len(quotes) - 1)

    tmp = re.sub(r"'[^']*'|\"[^\"]*\"|\u2018[^\u2019]*\u2019|\u201c[^\u201d]*\u201d",
                 _stash, text)
    parts = re.split(r"(?<=[.!?])\s+", tmp.strip())
    out = []
    for p in parts:
        for i, q in enumerate(quotes):
            p = p.replace("\x00%d\x00" % i, q)
        if p.strip():
            out.append(p)
    return out

def relevant_sentences(text, feature):
    """All sentences mentioning the feature (lower-cased)."""
    res = [s.lower() for s in split_sentences(text) if mentions(s.lower(), feature)]
    return res

def clause_polarity(clause):
    p = sum(clause.count(w) for w in POS_WORDS)
    n = sum(clause.count(w) for w in NEG_WORDS)
    return p, n

# Core: classify the feature's role inside ONE sentence
def role_in_sentence(sent, feature):
    if not mentions(sent, feature):
        return None
    low = sent
    fpos = first_pos(low, feature)
    has_causal = (any(c in low for c in SUPPORTING_MARKERS)
                  or any(p in low for p in SPANNING_PARTICIPLES))

    # comma-'and' / semicolon clause narrowing (unless a spanning participle
    # reaches back over the feature)
    trailing_part = any(re.search(r",\s*" + p, low[fpos:]) for p in SPANNING_PARTICIPLES)
    if not trailing_part:
        parts = re.split(r",\s+and\s+|;\s+", low)
        if len(parts) > 1:
            for p in parts:
                if mentions(p, feature):
                    low = p
                    fpos = first_pos(low, feature)
                    has_causal = (any(c in low for c in SUPPORTING_MARKERS)
                                  or any(p in low for p in SPANNING_PARTICIPLES))
                    break

    # explicit neutral ("... is/are (a) neutral") within the feature's clause
    mneu = re.search(r"\b(is|are)\s+(a\s+)?neutral", low)
    if mneu and fpos < mneu.start():
        between = low[fpos:mneu.start()]
        if (not re.search(r"\bwhile\b|;|,\s+and\s+", between)
                and not any(c in between for c in SUPPORTING_MARKERS)):
            return "neutral"

    # 'outweigh/override': verb-internal contrast
    mo = None
    for m in OUTWEIGH_MARKERS:
        i = low.find(m)
        if i != -1:
            mo = i if mo is None else min(mo, i)
    if mo is not None:
        return "support" if fpos < mo else "refuted"

    # 'while': additive (both sides causal & same polarity) vs. concessive
    wm = re.search(r"\bwhile\b", low)
    if wm:
        left, right = low[:wm.start()], low[wm.start():]
        lc = any(c in left for c in SUPPORTING_MARKERS)
        rc = any(c in right for c in SUPPORTING_MARKERS)
        if lc and rc:  # additive: narrow to the feature's own side
            low = left.rstrip(" ,") if fpos < wm.start() else right
            fp = first_pos(low, feature)
            if fp != -1:
                fpos = fp
            has_causal = (any(c in low for c in SUPPORTING_MARKERS)
                          or any(p in low for p in SPANNING_PARTICIPLES))
        else:          # concessive 'while X, Y': X (before) is refuted side
            if fpos < wm.start():
                return "refuted"
            return "support" if has_causal else "neutral"

    # concessive markers (despite/although/...): after-clause is refuted
    for m in CONCESSIVE_MARKERS:
        idx = low.find(m)
        if idx != -1:
            after = low[idx:]
            comma = after.find(",")
            conc_span = (idx, idx + comma) if comma != -1 else (idx, len(low))
            in_conc = conc_span[0] <= fpos <= conc_span[1]
            return "refuted" if in_conc else "support"

    # adversative markers (but/however/whereas): before-clause is refuted
    adv_idx = -1
    for m in ADVERSATIVE_MARKERS:
        i = low.find(m)
        if i != -1:
            adv_idx = i if adv_idx == -1 else min(adv_idx, i)
    if adv_idx != -1:
        left, right = low[:adv_idx], low[adv_idx:]
        left_causal = any(c in left for c in SUPPORTING_MARKERS)
        right_causal = any(c in right for c in SUPPORTING_MARKERS)
        if fpos < adv_idx:
            return "support" if (left_causal and not right_causal) else "refuted"
        else:
            return "support" if right_causal or not left_causal else "neutral"

    # no contrast
    if not has_causal:
        return "neutral"

    # spanning trailing participle attributes an effect to the feature
    if any(p in low[fpos:] for p in SPANNING_PARTICIPLES):
        return "support"

    # true accompaniment 'with' (not a complement like 'engagement with')
    window = low[max(0, fpos - 20):fpos]
    complement = re.search(r"(" + "|".join(WITH_COMPLEMENT_NOUNS) + r")\s+with\s*$", window)
    if not complement and any(a.strip() in window for a in [" with ", "along with", "as well as"]):
        return "neutral"

    # segment analysis: does the causal verb govern the feature or another one?
    segs = re.split(r",| and ", low)
    causal_segs = [s for s in segs if any(c in s for c in SUPPORTING_MARKERS)]
    feat_seg = next((s for s in segs if mentions(s, feature)), None)
    if feat_seg is not None:
        if any(c in feat_seg for c in SUPPORTING_MARKERS):
            return "support"
        # copula fact ("is large", "is Finance") -> neutral if verb belongs elsewhere
        if re.search(r"(is|are|was|were)\s+(large|small|relatively|high|low|short|"
                     r"average|numerous|very|promising|relevant|targeted|strong|"
                     r"positive|medium|finance|healthcare|energy|retail|manufacturing)",
                     feat_seg):
            # unless another feature shares the same later causal verb as coordinated subject
            cpos_list = [low.find(c) for c in SUPPORTING_MARKERS if c in low and low.find(c) > fpos]
            if cpos_list:
                cpos = min(cpos_list)
                own_copula = re.search(r"\b(is|are|was|were)\b", low[fpos:cpos])
                if own_copula:
                    return "neutral"
            return "neutral"
        # bare value / coordinated subject sharing a later causal verb
        cpos_list = [low.find(c) for c in SUPPORTING_MARKERS if c in low and low.find(c) > fpos]
        if cpos_list:
            cpos = min(cpos_list)
            between = low[fpos:cpos]
            if not any(a.strip() in between for a in [" with ", "along with", "as well as"]):
                own_copula = re.search(r"\b(is|are|was|were)\b", between)
                pre = low[max(0, cpos - 45):cpos]
                other_subject = any(
                    any(t in pre for t in FEATURE_TERMS[of]) for of in FEATURE_TERMS if of != feature
                )
                if other_subject and own_copula:
                    return "neutral"
                return "support"
        if causal_segs:
            return "neutral"
    return "support" if has_causal else "neutral"

# Suggest a code 0..3 for the feature in the whole explanation
def suggest_code(text, feature):
    if feature in ("BASE", "unknown"):
        return "", ""
    low = str(text).lower()
    if not mentions(low, feature):
        return 0, "not mentioned"

    roles = []
    for sent in relevant_sentences(text, feature):
        r = role_in_sentence(sent, feature)
        if r:
            roles.append(r)
    if not roles:
        return 0, "not mentioned"

    if "support" in roles:
        return 3, "supporting / causal"
    if "refuted" in roles:
        return 2, "refuted side of contrast"
    return 1, "neutral / enumeration"

# Build the coding template
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
        "final_code": code,   # Author overwrites this column during review
        "checked": "",        # Author marks with X once reviewed
    })

coding = pd.DataFrame(rows)

# Drop base rows since there is no perturbed feature to code
coding_to_review = coding[~coding["perturbed_feature"].isin(["BASE", "unknown"])].copy()

# Save the review file
coding_to_review.to_csv(BASE_DIR / "llm_lead_intent_results_explanations_coded.csv", index=False)

# Short summary
print("rows total:", len(coding))
print("rows to review:", len(coding_to_review))
print("\nsuggested code distribution:")
print(coding_to_review["suggested_code"].value_counts(dropna=False).sort_index())
print("\nby feature:")
print(coding_to_review.groupby("perturbed_feature")["suggested_code"]
      .value_counts().unstack(fill_value=0))
