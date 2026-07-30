import spacy
import re
import os
import json
from groq import Groq
from database import get_verb_info
from models import (
    EvaluateResponse, ScoreBreakdown,
    ActionScore, KnowledgeScore, ConditionScore, CriteriaScore,
)

# Load spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: en_core_web_sm not found. Ensure you ran python -m spacy download en_core_web_sm")
    nlp = spacy.blank("en")

# Configure Groq client (requires GROQ_API_KEY environment variable)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# Regex patterns for rule-based detection
# ---------------------------------------------------------------------------

CONDITION_PATTERNS = [
    r"\busing\b",
    r"\bthrough\b",
    r"\bgiven\b",
    r"\bwithout\b",
    r"\bunder\b.*\bconditions?\b",
    r"\bwith the help of\b",
    r"\bby means of\b",
    r"\bafter\b",
    r"\bprovided\b",
    r"\bvia\b",
    r"\bby\b\s+\w+ing\b",          # "by applying...", "by performing..."
]

CRITERIA_PATTERNS = [
    r"\b\d+\s*%\b",                 # "90%"
    r"\baccurate\s+to\b",          # "accurate to second decimal"
    r"\baccuracy\b",
    r"\bwithin\b.*\b(?:minutes?|hours?|seconds?|days?|attempts?)\b",
    r"\bno more than\b",
    r"\bat least\b",
    r"\bwith(?:out)?\s+\d+\s+errors?\b",
    r"\bcorrectly\b",
    r"\bsuccessfully\b",
]

# Patterns used to trim knowledge phrase — stop extraction at the first match
KNOWLEDGE_BOUNDARY_PATTERNS = [
    r"\busing\b",
    r"\bthrough\b",
    r"\bgiven\b",
    r"\bwithout\b",
    r"\bunder\b",
    r"\bwith the help of\b",
    r"\bby means of\b",
    r"\bvia\b",
    r"\bby\b\s+\w+ing\b",
    r"\bafter\b",
    r"\bprovided\b",
    r"\baccurate\s+to\b",
    r"\baccurate\b",
    r"\baccuracy\b",
    r"\bwith\s+\d",
    r"\bwithin\b",
    r"\bno more than\b",
    r"\bat least\b",
    r"\bcorrectly\b",
    r"\bsuccessfully\b",
    r",\s*accurate\b",
    r",\s*using\b",
    r",\s*with\b",
    r",\s*within\b",
]

VAGUE_KNOWLEDGE_WORDS = [
    r"\blike\b",
    r"\bsuch as\b",
    r"\bdifferent\b",
    r"\bvarious\b",
    r"\betc\.?\b",
]

VAGUE_VERBS = {"know", "understand", "learn", "appreciate", "study"}


def extract_main_verb(sentence: str) -> str:
    """Extract the main (root) verb lemma from a sentence using spaCy."""
    doc = nlp(sentence)
    # Prefer ROOT verb
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            return token.lemma_.lower()
    # Fallback: first verb token
    for token in doc:
        if token.pos_ == "VERB":
            return token.lemma_.lower()
    # Heuristic for "Students will [verb]" patterns
    words = sentence.split()
    if len(words) > 2 and words[0].lower() in ["students", "learners"]:
        return words[2].lower()
    return ""


def count_action_verbs(sentence: str) -> list[str]:
    """Return list of all verb lemmas found in the sentence (excluding auxiliaries)."""
    doc = nlp(sentence)
    verbs = []
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ not in ("aux", "auxpass"):
            verbs.append(token.lemma_.lower())
    return verbs


def extract_knowledge_phrase(sentence: str, verb: str) -> str:
    """Extract the direct object / complement of the main verb as the knowledge
    element, trimming at the first condition or criteria marker."""
    # Strategy: find everything after the verb, then trim at the earliest
    # condition/criteria boundary pattern.
    lower = sentence.lower()
    idx = lower.find(verb)
    if idx < 0:
        return ""

    after = sentence[idx + len(verb):].strip()
    if not after:
        return ""

    # Find the earliest boundary marker
    earliest_pos = len(after)
    for pat in KNOWLEDGE_BOUNDARY_PATTERNS:
        m = re.search(pat, after, re.IGNORECASE)
        if m and m.start() < earliest_pos:
            earliest_pos = m.start()

    knowledge = after[:earliest_pos].strip().rstrip(",;. ")
    return knowledge if knowledge else ""


def build_full_text(sentence: str,
                    condition: str | None = None,
                    criteria: str | None = None) -> str:
    """Combine sentence + optional condition/criteria into one string for
    condition/criteria scanning."""
    parts = [sentence.strip()]
    if condition and condition.strip():
        parts.append(condition.strip())
    if criteria and criteria.strip():
        parts.append(criteria.strip())
    return " ".join(parts)


def rule_based_check(sentence: str, full_text: str) -> dict:
    """Run rule-based NLP checks.

    - Action verb / knowledge: extracted from `sentence` only.
    - Condition / criteria regex: scanned from `full_text` (sentence + any
      user-provided condition/criteria boxes combined).
    """
    verb = extract_main_verb(sentence)
    verb_info = get_verb_info(verb) if verb else None
    all_verbs = count_action_verbs(sentence)
    knowledge_phrase = extract_knowledge_phrase(sentence, verb) if verb else ""

    # Condition detection (regex on full_text)
    condition_detected = False
    for pattern in CONDITION_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            condition_detected = True
            break

    # Criteria detection (regex on full_text)
    criteria_detected = False
    for pattern in CRITERIA_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            criteria_detected = True
            break

    # Vague knowledge words (check sentence only)
    vague_knowledge_flags = []
    for pattern in VAGUE_KNOWLEDGE_WORDS:
        if re.search(pattern, sentence, re.IGNORECASE):
            vague_knowledge_flags.append(pattern.replace(r"\b", "").replace("\\", ""))

    is_vague_verb = verb in VAGUE_VERBS

    return {
        "verb": verb,
        "verb_info": verb_info,
        "all_verbs": all_verbs,
        "verb_count": len(all_verbs),
        "is_vague_verb": is_vague_verb,
        "knowledge_phrase": knowledge_phrase,
        "vague_knowledge_flags": vague_knowledge_flags,
        "condition_detected": condition_detected,
        "criteria_detected": criteria_detected,
    }


def call_llm_for_evaluation(sentence: str, full_text: str,
                             rule_findings: dict,
                             condition_input: str | None = None,
                             criteria_input: str | None = None) -> dict:
    """Send the sentence + full_text to Groq LLM for structured evaluation.

    - Action / Knowledge: scored from the main `sentence` only (knowledge
      trimmed at condition/criteria markers).
    - Condition / Criteria: scored from the `full_text` (combined string).
      If a separate condition/criteria box was provided, prioritise that text.
    """
    if not groq_client:
        return {"error": "Groq API key missing."}

    has_condition_box = bool(condition_input and condition_input.strip())
    has_criteria_box = bool(criteria_input and criteria_input.strip())

    # Build dynamic condition section
    if has_condition_box:
        condition_section = f"""The user provided a SEPARATE condition field: "{condition_input.strip()}"
Validate whether this is genuinely a condition (method/tool/circumstance).
- Valid condition → score 0–2 based on specificity, set detected=true, condition_text = user's text.
- Not a valid condition → score 0, detected=false, explain why."""
    elif rule_findings["condition_detected"]:
        condition_section = f"""The user did NOT fill the separate condition box, BUT a condition phrase was detected in the main sentence.
Full text to scan: "{full_text}"
Extract the condition phrase from within this text.
- If a specific method/tool/circumstance is named → score 1–2, detected=true, set condition_text to the extracted phrase.
- If the condition is vague → score 1, detected=true."""
    else:
        condition_section = """No condition was provided in the separate box, and no condition phrase was detected in the sentence.
Score 0, detected=false, condition_text=null.
Feedback: "No condition provided — consider specifying the method, tool, or circumstance (e.g., 'using Newton-Raphson method through C++')." """

    # Build dynamic criteria section
    if has_criteria_box:
        criteria_section = f"""The user provided a SEPARATE criteria field: "{criteria_input.strip()}"
Validate whether this is genuinely a measurable standard.
- Valid & specific → score 2, detected=true, criteria_text = user's text.
- Valid but vague (e.g. "correctly") → score 1, detected=true.
- Not valid → score 0, detected=false, explain why."""
    elif rule_findings["criteria_detected"]:
        criteria_section = f"""The user did NOT fill the separate criteria box, BUT a criteria phrase was detected in the main sentence.
Full text to scan: "{full_text}"
Extract the criteria phrase from within this text.
- If a specific measurable standard is present → score 1–2, detected=true, set criteria_text to the extracted phrase.
- If vague → score 1, detected=true."""
    else:
        criteria_section = """No criteria was provided in the separate box, and no criteria phrase was detected in the sentence.
Score 0, detected=false, criteria_text=null.
Feedback: "No criteria provided — consider specifying a measurable standard (e.g., 'with 90% accuracy', 'accurate to second decimal place')." """

    prompt = f"""You are an expert instructional designer specializing in Course Outcome (CO) evaluation using Bloom's Taxonomy. Evaluate the following learning objective.

The user provides:
1. A main sentence (REQUIRED) — contains at minimum the Action verb + Knowledge element, and MAY ALSO contain condition/criteria phrases embedded within it.
2. An optional separate Condition field
3. An optional separate Criteria field

The full combined text (sentence + any separate fields) is:
"{full_text}"

Main sentence alone: "{sentence}"
{f'Separate condition box: "{condition_input.strip()}"' if has_condition_box else 'Separate condition box: (empty)'}
{f'Separate criteria box: "{criteria_input.strip()}"' if has_criteria_box else 'Separate criteria box: (empty)'}

## Rule-based NLP findings
- Extracted main verb: {rule_findings['verb']}
- Bloom's taxonomy DB lookup: {rule_findings['verb_info']}
- All verbs found: {rule_findings['all_verbs']} (count: {rule_findings['verb_count']})
- Is vague verb: {rule_findings['is_vague_verb']}
- Extracted knowledge phrase (trimmed at condition/criteria markers): "{rule_findings['knowledge_phrase']}"
- Vague knowledge words detected: {rule_findings['vague_knowledge_flags']}
- Condition detected in full text (regex): {rule_findings['condition_detected']}
- Criteria detected in full text (regex): {rule_findings['criteria_detected']}

## Scoring Rubric (total: 10 points)

### 1. Action / Bloom's Level (0–4 points) — from main sentence
Score Action deterministically as the SUM of two sub-components:

a) verb_validity_score (0–2 points):
- 2 pts: Exactly one valid action verb present (or two genuinely co-equal verbs applying to the same knowledge element), matched successfully in the Bloom's verb table.
- 1 pt: Verb present but ambiguous, borderline vague, or more than 2 verbs used.
- 0 pts: No clear action verb, or verb is non-demonstrable ("know", "understand" alone, "learn", "appreciate", "study").

b) bloom_weight_score (0–2 points):
- 2 pts: Verb maps to Apply, Analyze, Evaluate, or Create (higher-order thinking).
- 1 pt: Verb maps to Understand.
- 0 pts: Verb maps to Remember, or no verb found.

Action score MUST BE the exact sum: verb_validity_score + bloom_weight_score (0–4 points).

### 2. Knowledge Specificity (0–2 points) — from main sentence, TRIMMED at condition/criteria markers
The knowledge element is the DIRECT OBJECT of the action verb — what is being acted on.
IMPORTANT: Knowledge must NOT include condition phrases (using..., through...) or criteria phrases (accurate to..., with X% accuracy). Stop at the first such marker.
For example in "Determine the root of the given equation using Newton-Raphson method", the knowledge is ONLY "the root of the given equation", NOT "the root of the given equation using Newton-Raphson method".
- 0: Missing or extremely vague
- 1: Present but uses vague words (like, such as, various, different, etc.)
- 2: Concrete, specific knowledge element

### 3. Condition (0–2 points) — from full combined text, prioritizing separate box
{condition_section}

### 4. Criteria (0–2 points) — from full combined text, prioritizing separate box
{criteria_section}

## Checklist (flag in action/knowledge feedback)
- Does the sentence begin with / center on an action verb?
- Is it student performance (not teacher activity)?
- Is it a demonstrable product (not a process)?
- Does it avoid vague enumeration words?

## Few-Shot Examples

Example 1 — all in one sentence (10/10):
Main sentence: "Determine the root of the given equation, accurate to second decimal place, using Newton-Raphson method through C++ programming language."
Separate condition box: (empty)
Separate criteria box: (empty)
→ Action: Determine (verb_validity_score=2, bloom_weight_score=2 → score=4/4) | Knowledge: "root of the given equation" (2/2) | Condition: extracted "using Newton-Raphson method through C++ programming language" (2/2) | Criteria: extracted "accurate to second decimal place" (2/2) = Total: 10/10.

Example 2 — split across fields (10/10):
Main sentence: "Determine the root of the given equation"
Separate condition box: "using Newton-Raphson method through C++ programming language"
Separate criteria box: "accurate to second decimal place"
→ Same scores as Example 1. Both approaches yield 10/10.

Example 3 — partial split (10/10):
Main sentence: "Determine the root of the given equation using Newton-Raphson method through C++ programming language"
Separate condition box: (empty)
Separate criteria box: "accurate to second decimal place"
→ Action: Determine (verb_validity_score=2, bloom_weight_score=2 → score=4/4) | Knowledge: "root of the given equation" (2/2) | Condition: extracted from sentence (2/2) | Criteria: from box (2/2) = Total: 10/10.

Example 4 — no optional elements (6/10):
Main sentence: "Analyze the frequency response of a second-order control system"
→ Action: Analyze (verb_validity_score=2, bloom_weight_score=2 → score=4/4) | Knowledge: specific (2/2) | Condition: 0 | Criteria: 0 = Total: 6/10.

## Output

For "improved_objective": ALWAYS synthesize action + knowledge + condition + criteria into ONE complete, well-formed 10/10 sentence. If condition/criteria were absent, invent appropriate ones.

Return ONLY a valid JSON object (no markdown, no extra text):
{{
  "total_score": 10,
  "action": {{
    "verb_validity_score": 2,
    "bloom_weight_score": 2,
    "score": 4,
    "bloom_level": "Apply",
    "verb": "determine",
    "feedback": "The action verb 'determine' is a valid, single verb mapped to the Apply level of Bloom's Taxonomy."
  }},
  "knowledge": {{
    "score": 2,
    "detected_knowledge": "root of the given equation",
    "feedback": "Short feedback."
  }},
  "condition": {{
    "score": 2,
    "detected": true,
    "condition_text": "using Newton-Raphson method through C++",
    "feedback": "Short feedback."
  }},
  "criteria": {{
    "score": 2,
    "detected": true,
    "criteria_text": "accurate to second decimal place",
    "feedback": "Short feedback."
  }},
  "overall_feedback": "Overall summary.",
  "improved_objective": "A single, complete 10/10 sentence."
}}
"""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        response_text = chat_completion.choices[0].message.content.strip()
        return json.loads(response_text)
    except Exception as e:
        print("Failed to call LLM or parse JSON:", str(e))
        raise ValueError(f"LLM Error: {str(e)}")


def evaluate_sentence(sentence: str,
                      condition: str | None = None,
                      criteria: str | None = None) -> EvaluateResponse:
    """Full evaluation pipeline: rule-based extraction → LLM scoring → response assembly."""
    # Build combined full_text for condition/criteria scanning
    full_text = build_full_text(sentence, condition, criteria)

    # 1. Rule-based extraction
    #    - verb/knowledge from sentence only
    #    - condition/criteria regex from full_text
    rules = rule_based_check(sentence, full_text)

    # 2. LLM evaluation
    llm_result = call_llm_for_evaluation(
        sentence, full_text, rules,
        condition_input=condition,
        criteria_input=criteria,
    )

    if "error" in llm_result:
        # Fallback when API key is missing
        verb_info = rules["verb_info"]
        level = verb_info["taxonomy_level"] if verb_info else "Unknown"
        has_cond = rules["condition_detected"]
        has_crit = rules["criteria_detected"]
        return EvaluateResponse(
            total_score=0,
            breakdown=ScoreBreakdown(
                action=ActionScore(score=0, bloom_level=level, verb=rules["verb"], feedback="API Key Missing"),
                knowledge=KnowledgeScore(score=0, detected_knowledge=rules["knowledge_phrase"], feedback="API Key Missing"),
                condition=ConditionScore(
                    score=0, detected=has_cond,
                    condition_text=condition.strip() if condition and condition.strip() else None,
                    feedback="API Key Missing",
                ),
                criteria=CriteriaScore(
                    score=0, detected=has_crit,
                    criteria_text=criteria.strip() if criteria and criteria.strip() else None,
                    feedback="API Key Missing",
                ),
            ),
            overall_feedback="Please configure your GROQ_API_KEY environment variable to enable full scoring.",
            improved_objective="Please configure your GROQ_API_KEY environment variable to see an improved objective.",
        )

    # 3. Validation and Score Assertion for Action Component
    action_data = llm_result.get("action", {})
    v_val = action_data.get("verb_validity_score")
    b_weight = action_data.get("bloom_weight_score")

    if v_val is not None and b_weight is not None:
        computed_action_score = v_val + b_weight
        if action_data.get("score") != computed_action_score:
            print(f"[Scorer Warning] LLM returned action score {action_data.get('score')} but sub-components sum to {computed_action_score} (validity: {v_val}, weight: {b_weight}). Overriding score to {computed_action_score}.")
            action_data["score"] = computed_action_score

    # Check for sentiment consistency in action feedback
    feedback_lower = action_data.get("feedback", "").lower()
    negative_keywords = {"could", "consider", "however", "but", "vague", "weak", "unclear", "lack", "lacks", "missing", "ambiguous", "non-demonstrable", "should", "needs"}
    has_negative = any(kw in feedback_lower for kw in negative_keywords)
    bloom_level = action_data.get("bloom_level", "")
    higher_order_levels = {"Apply", "Analyze", "Evaluate", "Create"}

    if not has_negative and bloom_level in higher_order_levels and action_data.get("score", 0) < 4:
        print(f"[Scorer Warning] Mismatched Action score ({action_data.get('score')}/4) for '{bloom_level}' verb with feedback lacking negative sentiment: '{action_data.get('feedback')}'. Correcting score to 4.")
        action_data["score"] = 4
        action_data["verb_validity_score"] = 2
        action_data["bloom_weight_score"] = 2

    # Re-calculate total score
    act_s = action_data.get("score", 0)
    kno_s = llm_result.get("knowledge", {}).get("score", 0)
    con_s = llm_result.get("condition", {}).get("score", 0)
    cri_s = llm_result.get("criteria", {}).get("score", 0)
    llm_result["total_score"] = min(10, act_s + kno_s + con_s + cri_s)

    # 4. Assemble response from LLM JSON
    return EvaluateResponse(
        total_score=llm_result["total_score"],
        breakdown=ScoreBreakdown(
            action=ActionScore(**action_data),
            knowledge=KnowledgeScore(**llm_result["knowledge"]),
            condition=ConditionScore(**llm_result["condition"]),
            criteria=CriteriaScore(**llm_result["criteria"]),
        ),
        overall_feedback=llm_result["overall_feedback"],
        improved_objective=llm_result.get("improved_objective", "Could not generate improved objective."),
    )
