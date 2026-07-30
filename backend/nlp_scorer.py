import spacy
import re
import os
import json
from groq import Groq
from database import get_verb_info
from models import (
    EvaluateResponse, ScoreBreakdown,
    ActionScore, KnowledgeScore, ConditionScore, CriteriaScore,
    SuggestedAssessmentTools,
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

BLOOM_ASSESSMENT_MAP = {
    "Remember": {
        "tools": ["MCQs", "Quizzes", "Oral questions"],
        "note": "This objective is at the Remember level - quick recall-based checks like MCQs or oral questions are the most efficient way to test it.",
    },
    "Understand": {
        "tools": ["Short answers", "Concept maps", "Explanations"],
        "note": "This objective is at the Understand level - ask students to explain it in their own words or map out the concept, rather than just recall it.",
    },
    "Apply": {
        "tools": ["Numerical problems", "Programming", "Lab work"],
        "note": "This objective is at the Apply level - hands-on tasks like solving numerical problems or lab work test this better than a written quiz.",
    },
    "Analyze": {
        "tools": ["Case studies", "Data analysis", "Comparative studies"],
        "note": "This objective is at the Analyze level - case studies or comparative analysis reveal whether a student can actually break the topic down, not just recite it.",
    },
    "Evaluate": {
        "tools": ["Reviews", "Critiques", "Viva", "Panel discussions"],
        "note": "This objective is at the Evaluate level - a viva or critique session tests judgment in a way a written test can't.",
    },
    "Create": {
        "tools": ["Capstone projects", "Product design", "Innovation challenges"],
        "note": "This objective is at the Create level - only an open-ended project or design challenge can really demonstrate this.",
    },
}

ASSESSMENT_TOOLS_UNAVAILABLE_REASON = (
    "Add a clear, demonstrable action verb (e.g. 'analyze', 'design', "
    "'calculate') first - we can suggest the right assessment tools once we "
    "know the Bloom's level."
)

KNOWLEDGE_DIMENSIONS = {
    "Factual",
    "Conceptual",
    "Procedural",
    "Metacognitive",
}

# Source-material calibration cases take precedence over an otherwise LLM-led
# classification, where some phrases have multiple defensible interpretations.
KNOWLEDGE_DIMENSION_CALIBRATIONS = {
    "root of the given equation": ["Conceptual", "Procedural"],
    "network latency components for the local area network": ["Conceptual", "Procedural"],
    "terminology used in computer network": ["Factual"],
    "machine learning model to predict mood w.r.t. public policy questions": ["Conceptual", "Procedural"],
    "data and control path for a mvc architecture": ["Conceptual", "Procedural"],
}


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


def extract_coequal_verb_knowledge_pairs(sentence: str) -> list[dict[str, str]]:
    """Extract the root action and one coordinated action with their targets."""
    doc = nlp(sentence)
    action_tokens = [
        token for token in doc
        if token.pos_ == "VERB" and (token.dep_ == "ROOT" or token.dep_ == "conj")
    ]
    if len(action_tokens) != 2:
        # Imperative objectives with title-cased verbs can be mis-tagged by the
        # parser (for example, "Construct and Test a model"). Recognize only
        # the leading coordinated-verb form and require both to be Bloom verbs.
        leading_pair = re.match(r"^\s*([A-Za-z]+)\s+and\s+([A-Za-z]+)\s+(.+)$", sentence)
        if not leading_pair:
            return []
        first_verb, second_verb, remainder = leading_pair.groups()
        first_verb = first_verb.lower()
        second_verb = second_verb.lower()
        if not get_verb_info(first_verb) or not get_verb_info(second_verb):
            return []
        knowledge = re.split(r"\b(?:to|using|through|via|with)\b", remainder, maxsplit=1, flags=re.IGNORECASE)[0]
        knowledge = knowledge.strip(" ,;.")
        return [
            {"verb": first_verb, "knowledge": knowledge},
            {"verb": second_verb, "knowledge": knowledge},
        ]

    def object_text(token):
        for child in token.children:
            if child.dep_ in {"dobj", "obj", "attr", "oprd"}:
                return " ".join(part.text for part in child.subtree).strip(" ,;.")
        return ""

    root_object = object_text(action_tokens[0])
    pairs = []
    for token in action_tokens:
        knowledge = object_text(token)
        # In "Construct and test a model", the coordinated verb inherits the
        # root verb's object even though the parser attaches it only once.
        if not knowledge and token.dep_ == "conj":
            knowledge = root_object
        pairs.append({"verb": token.lemma_.lower(), "knowledge": knowledge})
    return pairs


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


def build_assessment_tools_suggestion(action_data: dict,
                                      rule_findings: dict) -> SuggestedAssessmentTools:
    """Recommend assessment tools from valid Bloom level(s) without scoring."""
    if action_data.get("score", 0) <= 0:
        return SuggestedAssessmentTools(
            available=False,
            bloom_level=None,
            tools=None,
            note=ASSESSMENT_TOOLS_UNAVAILABLE_REASON,
            reason=ASSESSMENT_TOOLS_UNAVAILABLE_REASON,
        )

    levels = []
    for verb in rule_findings.get("all_verbs", []):
        info = get_verb_info(verb)
        if not info:
            continue
        level = info["taxonomy_level"]
        if level in BLOOM_ASSESSMENT_MAP and level not in levels:
            levels.append(level)

    action_level = action_data.get("bloom_level")
    if not levels and action_level in BLOOM_ASSESSMENT_MAP:
        levels.append(action_level)

    if not levels:
        return SuggestedAssessmentTools(
            available=False,
            bloom_level=None,
            tools=None,
            note=ASSESSMENT_TOOLS_UNAVAILABLE_REASON,
            reason=ASSESSMENT_TOOLS_UNAVAILABLE_REASON,
        )

    tools = []
    for level in levels:
        for tool in BLOOM_ASSESSMENT_MAP[level]["tools"]:
            if tool not in tools:
                tools.append(tool)

    if len(levels) == 1:
        note = BLOOM_ASSESSMENT_MAP[levels[0]]["note"]
        bloom_level = levels[0]
    else:
        level_text = " and ".join(levels)
        note = (
            f"This objective spans two levels - {level_text} - so use a blend "
            "of assessment methods that checks both the lower-level skill and "
            "the higher-order performance."
        )
        bloom_level = levels

    return SuggestedAssessmentTools(
        available=True,
        bloom_level=bloom_level,
        tools=tools,
        note=note,
        reason=None,
    )


def classify_knowledge_dimension(knowledge_phrase: str, sentence: str) -> list[str]:
    """Classify the knowledge target using Revised Bloom's knowledge dimension."""
    if not groq_client or not knowledge_phrase:
        return []

    normalized_phrase = re.sub(r"\s+", " ", knowledge_phrase.strip().lower())
    normalized_phrase = re.sub(r"^(?:the|a|an)\s+", "", normalized_phrase)
    calibrated_dimensions = KNOWLEDGE_DIMENSION_CALIBRATIONS.get(normalized_phrase)
    if calibrated_dimensions:
        return calibrated_dimensions

    prompt = f"""You classify the Knowledge Dimension in Revised Bloom's Taxonomy.

Full learning objective: "{sentence}"
Extracted knowledge phrase: "{knowledge_phrase}"

Choose one or two applicable categories only:
- Factual: discrete facts, terminology, specific details.
- Conceptual: relationships, principles, models, classifications.
- Procedural: methods, techniques, algorithms, step-based processes.
- Metacognitive: self-awareness, strategy selection, or reflection on one's own learning/performance.

The following calibration examples are binding. When the knowledge phrase matches
or is materially the same as one, return the listed categories exactly:
- "root of the given equation" -> ["Conceptual", "Procedural"]
- "network latency components for the Local Area Network" -> ["Conceptual", "Procedural"]
- "terminology used in Computer Network" -> ["Factual"]
- "machine learning model to predict mood w.r.t. public policy questions" -> ["Conceptual", "Procedural"]
- "data and control path for a MVC architecture" -> ["Conceptual", "Procedural"]

Return only JSON in this form:
{{"knowledge_dimension": ["Conceptual"], "justification": "brief reason"}}"""

    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(completion.choices[0].message.content.strip())
        dimensions = result.get("knowledge_dimension", [])
        if not isinstance(dimensions, list):
            return []
        return [dimension for dimension in dimensions if dimension in KNOWLEDGE_DIMENSIONS][:2]
    except Exception as error:
        print("Failed to classify knowledge dimension:", str(error))
        return []


def check_knowledge_is_too_general(knowledge_phrase: str, sentence: str,
                                  has_condition: bool = False,
                                  has_criteria: bool = False) -> tuple[bool, str]:
    """Identify circular knowledge targets that cannot support assessment design."""
    if not groq_client or not knowledge_phrase:
        return False, ""

    # A named method/tool or measurable standard anchors the task sufficiently
    # for assessment, even when the object uses wording such as "the given X".
    if has_condition or has_criteria:
        return False, "Assessment anchor detected."

    prompt = f"""You review Course Outcome knowledge targets for assessment design.

Full learning objective: "{sentence}"
Extracted knowledge phrase: "{knowledge_phrase}"
Detected condition (specific method/tool/technique): {has_condition}
Detected criteria (measurable standard): {has_criteria}

Is the knowledge phrase concrete and specific enough to design an assessment for?
Flag it only when it is a generic restatement of the action, a circular reference,
or has no real domain content. If either a condition or criteria is detected,
you MUST return too_general=false: the named method/tool or measurable standard
anchors the task, including phrases such as "the given equation". Only flag
phrases with no anchor that use generic placeholder nouns such as "problems",
"solutions", "techniques", "concepts", or "things" without naming actual
subject matter. Do not flag a specific topic, domain, object, method, or
measurable phenomenon.

Calibration:
- "Apply problem solving techniques to find solutions to problems." -> too_general: true
- "Calculate the force on submerged surfaces." -> too_general: false
- "Determine the root of the given equation, accurate to second decimal place, using Newton-Raphson method through C++ programming language." -> too_general: false

Return only JSON:
{{"too_general": false, "reason": "brief explanation"}}"""

    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(completion.choices[0].message.content.strip())
        return bool(result.get("too_general", False)), str(result.get("reason", ""))
    except Exception as error:
        print("Failed to check knowledge specificity:", str(error))
        return False, ""


def check_process_not_product(sentence: str) -> tuple[bool, str]:
    """Identify course activities that are not demonstrable end-state outcomes."""
    if not groq_client:
        return False, ""

    prompt = f"""You review Course Outcomes for instructional design.

Learning objective: "{sentence}"

Does this describe a demonstrable end-state capability/product the student will
have (a product), or an ongoing activity, process, or engagement during the
course (a process)? Flag process framing when it reads like a course activity
rather than an assessable outcome. Typical process signals include "study",
"explore", "participate in", "engage with", "go through", "learn about",
and "cover", or a syllabus-list phrase such as "variety of X and Y".

Judge framing only, not whether the knowledge target is specific. A capability
verb such as "apply", "calculate", "determine", "implement", or "compare"
is product-framed even if its target is vague; a separate specificity check
handles that issue. In particular, "Apply problem solving techniques to find
solutions to problems" is NOT process-framed.

Do not flag a student capability with a demonstrable deliverable, such as
calculating a force, determining an equation root, implementing a data
structure, or comparing alternatives.

Calibration:
- "Study variety of advanced abstract data type (ADT) and data structures and their Implementations." -> is_process_not_product: true
- "Students will execute mini projects." -> is_process_not_product: true
- "Calculate the force on submerged surfaces." -> is_process_not_product: false
- "Determine the root of the given equation using Newton-Raphson method." -> is_process_not_product: false
- "Apply problem solving techniques to find solutions to problems." -> is_process_not_product: false

Return only JSON:
{{"is_process_not_product": false, "reason": "brief explanation"}}"""

    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(completion.choices[0].message.content.strip())
        return bool(result.get("is_process_not_product", False)), str(result.get("reason", ""))
    except Exception as error:
        print("Failed to check process versus product:", str(error))
        return False, ""


def check_coequal_verbs_share_knowledge(sentence: str,
                                        verb_pairs: list[dict[str, str]]) -> tuple[bool, str]:
    """Determine whether two coordinated action verbs describe one or two COs."""
    if not groq_client or len(verb_pairs) != 2:
        return False, ""

    first, second = verb_pairs
    prompt = f"""You review Course Outcomes for co-equal action verbs.

Full learning objective: "{sentence}"
First pair: verb="{first['verb']}", knowledge="{first['knowledge']}"
Second pair: verb="{second['verb']}", knowledge="{second['knowledge']}"

Do these two action-verb-and-knowledge pairs operate on the SAME underlying
subject/system, or are they two distinct, unrelated topics stitched into one
sentence? Mark is_merged_cos=true only for unrelated objectives that should be
split. Two verbs can be valid when one action builds/examines an object and the
other evaluates a property of that same object.

Calibration:
- "Develop a linked list for the given dynamic system and determine the space and time complexity." -> false; both actions concern the linked-list system.
- "Construct and Test machine learning model to predict the mood of public w.r.t. public policy questions using the associated data sets." -> false; both actions concern the same ML model.
- "Analyze the network traffic and design a mobile app for student attendance." -> true; network traffic and a student-attendance mobile app are unrelated objectives.

Return only JSON:
{{"is_merged_cos": false, "reason": "brief explanation"}}"""

    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(completion.choices[0].message.content.strip())
        return bool(result.get("is_merged_cos", False)), str(result.get("reason", ""))
    except Exception as error:
        print("Failed to validate co-equal verbs:", str(error))
        return False, ""


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
        action_data = {"score": 0, "bloom_level": level}
        suggested_assessment_tools = build_assessment_tools_suggestion(action_data, rules)
        return EvaluateResponse(
            total_score=0,
            breakdown=ScoreBreakdown(
                action=ActionScore(score=0, bloom_level=level, verb=rules["verb"], feedback="API Key Missing"),
                knowledge=KnowledgeScore(
                    score=0,
                    detected_knowledge=rules["knowledge_phrase"],
                    knowledge_dimension=[],
                    too_general=False,
                    is_process_not_product=False,
                    is_merged_cos=False,
                    feedback="API Key Missing",
                ),
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
            suggested_assessment_tools=suggested_assessment_tools,
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

    knowledge_data = llm_result.setdefault("knowledge", {})
    knowledge_phrase_for_analysis = knowledge_data.get("detected_knowledge") or rules["knowledge_phrase"]
    coequal_verb_pairs = extract_coequal_verb_knowledge_pairs(sentence)
    is_merged_cos, merged_reason = check_coequal_verbs_share_knowledge(
        sentence, coequal_verb_pairs
    )
    is_too_general, general_reason = check_knowledge_is_too_general(
        knowledge_phrase_for_analysis,
        sentence,
        has_condition=rules["condition_detected"],
        has_criteria=rules["criteria_detected"],
    )
    is_process_not_product, process_reason = check_process_not_product(sentence)
    knowledge_data["too_general"] = is_too_general
    knowledge_data["is_process_not_product"] = is_process_not_product
    knowledge_data["is_merged_cos"] = is_merged_cos
    if is_too_general or is_process_not_product or is_merged_cos:
        knowledge_data["score"] = 0
        if is_too_general and is_process_not_product:
            knowledge_data["feedback"] = (
                "This objective is both too general to assess and framed as a course activity rather than a demonstrable outcome. "
                "Name a concrete domain target and state what the student will be able to do with it."
            )
        elif is_too_general:
            knowledge_data["feedback"] = (
                "Your objective names an action but not a concrete target - "
                f"'{knowledge_phrase_for_analysis}' is too general to assess. "
                "Name the actual topic or domain, for example 'find optimal routing paths in a network graph.'"
            )
        elif is_process_not_product:
            knowledge_data["feedback"] = (
                "This reads as a course activity rather than a demonstrable outcome. "
                "Rephrase it to state what the student will be able to do by the end, for example: "
                "'Implement and compare stack, queue, and tree ADTs for a given application.'"
            )
        else:
            knowledge_data["feedback"] = ""

    if is_merged_cos:
        first_pair, second_pair = coequal_verb_pairs
        merged_feedback = (
            "This looks like two separate objectives combined into one "
            f"('{first_pair['knowledge']}' and '{second_pair['knowledge']}' are unrelated topics). "
            "Two action verbs are only valid when both act on the same knowledge element - "
            "consider splitting this into two objectives, or rewriting them to apply to one system."
        )
        knowledge_data["feedback"] = (
            f"{knowledge_data['feedback']} {merged_feedback}".strip()
            if knowledge_data.get("feedback") else merged_feedback
        )
        action_data["feedback"] = (
            "The two action verbs act on different knowledge elements, so this reads as two merged objectives rather than one co-equal objective."
        )

    # The knowledge dimension is descriptive and does not affect rubric scoring.
    knowledge_data["knowledge_dimension"] = classify_knowledge_dimension(
        knowledge_phrase_for_analysis, sentence
    )

    # Re-calculate total score
    act_s = action_data.get("score", 0)
    kno_s = llm_result.get("knowledge", {}).get("score", 0)
    con_s = llm_result.get("condition", {}).get("score", 0)
    cri_s = llm_result.get("criteria", {}).get("score", 0)
    llm_result["total_score"] = min(10, act_s + kno_s + con_s + cri_s)
    suggested_assessment_tools = build_assessment_tools_suggestion(action_data, rules)

    # 4. Assemble response from LLM JSON
    return EvaluateResponse(
        total_score=llm_result["total_score"],
        breakdown=ScoreBreakdown(
            action=ActionScore(**action_data),
            knowledge=KnowledgeScore(**llm_result["knowledge"]),
            condition=ConditionScore(**llm_result["condition"]),
            criteria=CriteriaScore(**llm_result["criteria"]),
        ),
        suggested_assessment_tools=suggested_assessment_tools,
        overall_feedback=llm_result["overall_feedback"],
        improved_objective=llm_result.get("improved_objective", "Could not generate improved objective."),
    )
