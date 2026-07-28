from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

from models import EvaluateRequest, EvaluateResponse
from nlp_scorer import evaluate_sentence
from database import init_db

# Initialize database on startup
init_db()

app = FastAPI(title="Objective Sentence Scorer")

# Add CORS middleware for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_objective(request: EvaluateRequest):
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
