from pydantic import BaseModel
from typing import Optional

class EvaluateRequest(BaseModel):
    sentence: str

class CategoryScore(BaseModel):
    score: int
    feedback: str
    bloom_level: Optional[str] = None
    detected: Optional[bool] = None

class ScoreBreakdown(BaseModel):
    action_verb: CategoryScore
    condition: CategoryScore
    criterion: CategoryScore
    clarity: CategoryScore

class EvaluateResponse(BaseModel):
    total_score: int
    breakdown: ScoreBreakdown
    overall_feedback: str
    improved_objective: str
