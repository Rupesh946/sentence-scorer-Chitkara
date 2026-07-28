import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

from models import EvaluateRequest, EvaluateResponse
from nlp_scorer import evaluate_sentence
from database import init_db

# Initialize database on startup
init_db()

app = FastAPI(title="Objective Sentence Scorer")

# Restrict CORS to the deployed frontend URL in production.
# Set ALLOWED_ORIGIN env var in Vercel (e.g. https://your-app.vercel.app).
# Falls back to wildcard only when the variable is not set (local dev).
_allowed_origin = os.getenv("ALLOWED_ORIGIN", "*")
_origins = [_allowed_origin] if _allowed_origin != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest):
    if not request.sentence.strip():
        raise HTTPException(status_code=400, detail="Sentence cannot be empty.")

    try:
        response = evaluate_sentence(request.sentence)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
