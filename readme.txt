RAG PDF Assistant
==================

Upload any PDF — a resume, a research paper, a legal contract — ask a
question in plain English, and get a grounded answer back with the source
page number it came from.


How It Works
------------
1. Upload a PDF.
2. The backend extracts text, splits it into chunks, and creates embeddings.
3. Chunks are stored in a vector database.
4. When you ask a question, the query is embedded and matched against stored chunks.
5. The most relevant chunks are sent to an LLM along with your question.
6. You get an answer with citations pointing back to the source pages.


Active Structure
----------------
backend/                  → Python FastAPI service
backend/app/main.py       → App entrypoint — creates FastAPI instance, includes routers
backend/app/config.py     → Central settings (reads from environment variables)
backend/app/routes/       → API route modules (one file per feature)
backend/app/routes/health.py → Health-check endpoint (GET /health)
backend/requirements.txt  → Python dependencies
backend/.env.example      → Documents required environment variables

frontend/                 → UI (not started yet)

.env                      → Local secrets (gitignored, never committed)
.gitignore                → Files excluded from version control


Run Locally
-----------
1. Clone the repo:
   git clone <repo-url>
   cd rag-pdf-assistant

2. Create a virtual environment (recommended):
   python -m venv venv
   venv\Scripts\activate        (Windows)
   source venv/bin/activate     (Mac/Linux)

3. Install backend dependencies:
   pip install -r backend/requirements.txt

4. Set up environment variables:
   copy backend\.env.example .env   (Windows)
   cp backend/.env.example .env     (Mac/Linux)
   (edit .env and fill in your values)

5. Start the backend:
   cd backend
   python -m uvicorn app.main:app --reload --env-file ../.env

6. Open http://127.0.0.1:8000/health
   You should see: {"status": "ok", "environment": "development"}

7. View auto-generated API docs:
   http://127.0.0.1:8000/docs


Environment Variables
---------------------
Variable      Default                    Description
--------      -------                    -----------
APP_NAME      RAG PDF Assistant API      Application title shown in API docs
APP_ENV       development                Current environment (development / production)


Tech Stack (Current)
--------------------
- Python 3.12+
- FastAPI (web framework)
- Uvicorn (ASGI server)


Tech Stack (Planned)
--------------------
- PDF parsing: PyMuPDF or pdfplumber
- OCR: Tesseract or Azure Document Intelligence
- Embeddings: OpenAI text-embedding-3-large
- Vector DB: pgvector or Qdrant
- LLM: OpenAI GPT-4.1 / GPT-5
- Frontend: React or Next.js


Notes
-----
- The backend and frontend have separate dependency files.
- The .env file contains secrets and is never committed to git.
- Query execution will be restricted to read-only operations for safety.
- Each new feature gets its own route file in backend/app/routes/.