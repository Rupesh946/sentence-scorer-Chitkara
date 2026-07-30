from pydantic import BaseModel
from typing import Optional

class EvaluateRequest(BaseModel):
    sentence: str
    condition: Optional[str] = None
    criteria: Optional[str] = None

class ActionScore(BaseModel):
    score: int            # 0-4 (sum of verb_validity_score + bloom_weight_score)
    bloom_level: str      # Remember / Understand / Apply / Analyze / Evaluate / Create / Vague / Unknown
    verb: str             # detected action verb
    feedback: str
    verb_validity_score: Optional[int] = None
    bloom_weight_score: Optional[int] = None

class KnowledgeScore(BaseModel):
    score: int            # 0-2
    detected_knowledge: str   # what the verb acts on
    feedback: str

class ConditionScore(BaseModel):
    score: int            # 0-2
    detected: bool
    condition_text: Optional[str] = None
    feedback: str

class CriteriaScore(BaseModel):
    score: int            # 0-2
    detected: bool
    criteria_text: Optional[str] = None
    feedback: str

class ScoreBreakdown(BaseModel):
    action: ActionScore
    knowledge: KnowledgeScore
    condition: ConditionScore
    criteria: CriteriaScore

class EvaluateResponse(BaseModel):
    total_score: int            # 0-10
    breakdown: ScoreBreakdown
    overall_feedback: str
    improved_objective: str
