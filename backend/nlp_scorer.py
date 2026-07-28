import spacy
import re
import os
import json
from groq import Groq
from database import get_verb_info
from models import EvaluateResponse, ScoreBreakdown, CategoryScore

# Load spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback if not downloaded, but we expect it to be
    print("Warning: en_core_web_sm not found. Ensure you ran python -m spacy download en_core_web_sm")
    nlp = spacy.blank("en")

# Configure Groq client (requires GROQ_API_KEY environment variable)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

CONDITION_PATTERNS = [
    r"\bgiven\b",
    r"\busing\b",
    r"\bwithout\b",
    r"\bunder\b.*conditions",
    r"\bafter\b",
    r"\bprovided\b"
]

CRITERION_PATTERNS = [
    r"\b\d+%\b",
    r"\baccuracy\b",
    r"\bwithin\b.*\b(minutes|hours|seconds|days)\b",
    r"\bno more than\b",
    r"\bat least\b"
]

def extract_main_verb(sentence: str) -> str:
    doc = nlp(sentence)
    # A simple heuristic: find the first ROOT verb, or first verb.
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            return token.lemma_.lower()
    
    # Fallback to just the first verb if no ROOT verb found
    for token in doc:
        if token.pos_ == "VERB":
            return token.lemma_.lower()
            
    # If no verb tagged, just try to take the second word assuming "Students will [verb]" pattern
    # But usually spacy finds something.
    words = sentence.split()
    if len(words) > 2 and words[0].lower() in ["students", "learners"]:
        return words[2].lower()
    
    return ""

def rule_based_check(sentence: str):
    doc = nlp(sentence)
    verb = extract_main_verb(sentence)
    
    verb_info = get_verb_info(verb) if verb else None
    
    # Check for conditions
    has_condition = False
    for pattern in CONDITION_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            has_condition = True
            break
            
    # Check for criterion
    has_criterion = False
    for pattern in CRITERION_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            has_criterion = True
            break
            
    return {
        "verb": verb,
        "verb_info": verb_info,
        "has_condition": has_condition,
        "has_criterion": has_criterion
    }

def call_llm_for_evaluation(sentence: str, rule_findings: dict) -> dict:
    if not groq_client:
        return {
            "error": "Groq API key missing."
        }
        
    prompt = f"""You are an expert instructional designer. Evaluate the following learning objective sentence based on Bloom's Taxonomy and the ABCD model.
Sentence: "{sentence}"

Here is what a rule-based NLP system found:
- Extracted main verb: {rule_findings['verb']}
- Bloom's taxonomy info from DB: {rule_findings['verb_info']}
- Detected condition phrase via regex: {rule_findings['has_condition']}
- Detected criterion/degree phrase via regex: {rule_findings['has_criterion']}

Please provide a final scoring out of 10 based on this rubric:
1. Action verb (0-4 points):
   - 0: Missing or vague verb ("know", "understand", "learn")
   - 1-2: Lower order Bloom's (Remember, Understand)
   - 3-4: Higher order Bloom's (Apply, Analyze, Evaluate, Create)
   (Adjust based on your semantic understanding if the rule-based DB missed or misclassified the verb).
   *Feedback Instruction*: If the score is < 4, explicitly recommend a better verb.
2. Condition (0-2 points):
   - Does it specify the circumstances? (Confirm/override rule-based finding). 0 if absent, 2 if present.
   *Feedback Instruction*: If the score is 0, explicitly recommend adding a specific condition relevant to their sentence.
3. Criterion (0-2 points):
   - Does it specify a measurable standard? (Confirm/override rule-based finding). 0 if absent, 2 if present.
   *Feedback Instruction*: If the score is 0, explicitly recommend adding a specific criterion (e.g. "with 90% accuracy") relevant to their sentence.
4. Clarity & specificity (0-2 points):
   - Is the sentence clear, unambiguous, and focused on the learner? 0-2 points based on quality.
   *Feedback Instruction*: If < 2, recommend how to clarify it.
5. Provide a rewritten, perfect 10/10 example of this objective in the "improved_objective" field.

Return your evaluation ONLY as a valid JSON object matching this structure exactly (no markdown formatting, just raw JSON):
{{
  "total_score": 8,
  "action_verb": {{"score": 4, "bloom_level": "Analyze", "feedback": "Short feedback."}},
  "condition": {{"score": 2, "detected": true, "feedback": "Short feedback."}},
  "criterion": {{"score": 0, "detected": false, "feedback": "Short feedback."}},
  "clarity": {{"score": 2, "feedback": "Short feedback."}},
  "overall_feedback": "Overall summary.",
  "improved_objective": "Given a dataset, students will be able to analyze data using Excel with 90% accuracy."
}}
"""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        response_text = chat_completion.choices[0].message.content.strip()
        
        # Try to parse JSON from the response text
        return json.loads(response_text)
    except Exception as e:
        print("Failed to call LLM or parse JSON:", str(e))
        raise ValueError(f"LLM Error: {str(e)}")

def evaluate_sentence(sentence: str) -> EvaluateResponse:
    # 1. Rule-based
    rules = rule_based_check(sentence)
    
    # 2. LLM Evaluation
    llm_result = call_llm_for_evaluation(sentence, rules)
    
    if "error" in llm_result:
        # Provide a fallback mock response if API key is missing for UI development
        return EvaluateResponse(
            total_score=0,
            breakdown=ScoreBreakdown(
                action_verb=CategoryScore(score=0, feedback="API Key Missing", bloom_level=rules["verb_info"]["taxonomy_level"] if rules["verb_info"] else "Unknown"),
                condition=CategoryScore(score=0, feedback="API Key Missing", detected=rules["has_condition"]),
                criterion=CategoryScore(score=0, feedback="API Key Missing", detected=rules["has_criterion"]),
                clarity=CategoryScore(score=0, feedback="API Key Missing")
            ),
            overall_feedback="Please configure your GROQ_API_KEY environment variable to enable full scoring.",
            improved_objective="Please configure your GROQ_API_KEY environment variable to see an improved objective."
        )

    # 3. Combine and return
    return EvaluateResponse(
        total_score=llm_result["total_score"],
        breakdown=ScoreBreakdown(
            action_verb=CategoryScore(**llm_result["action_verb"]),
            condition=CategoryScore(**llm_result["condition"]),
            criterion=CategoryScore(**llm_result["criterion"]),
            clarity=CategoryScore(**llm_result["clarity"])
        ),
        overall_feedback=llm_result["overall_feedback"],
        improved_objective=llm_result.get("improved_objective", "Could not generate improved objective.")
    )
