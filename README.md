# Objective Sentence Scorer

A web application that evaluates learning-objective-style sentences based on Bloom's Taxonomy and the ABCD model.

## Prerequisites

- Python 3.9+
- Node.js 18+
- Gemini API Key

## Backend Setup

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: if `requirements.txt` is missing, you can run `pip install fastapi uvicorn spacy google-generativeai pydantic python-dotenv`)*

4. Download the spaCy english model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. Set up your environment variables:
   Copy `.env.example` to `.env` and add your Groq API key.
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```
   *(You can get a free Groq API key at https://console.groq.com/keys)*

6. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will be available at http://localhost:8000. 
   *(API docs at http://localhost:8000/docs)*

## Frontend Setup

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```
   The frontend will be available at http://localhost:5173.

## Usage

1. Open the frontend URL in your browser.
2. Enter a learning objective (e.g., "Students will be able to analyze data using Excel with 90% accuracy.").
3. Click "Score my sentence".
4. Review the /10 score and the feedback breakdown based on the ABCD rubric!
