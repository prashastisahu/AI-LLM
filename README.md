# AI Stylist
![CI](https://github.com/prashastisahu/AI-LLM/actions/workflows/ci.yml/badge.svg)
An AI fashion stylist. Upload an outfit photo, describe the occasion, and get
clothing/accessory suggestions matched to real European clothing (H&M) — not
just AI-invented text, but actual products with real images.

## What it does
- Analyses outfit photos with Gemini Vision and suggests clothing/accessory
  pieces to complete the look (`POST /style`)
- Matches each suggestion to a real product from a Postgres-backed catalogue of
  H&M clothing (semantic search via ChromaDB), when one exists
- Serves that catalogue directly too: filter by category/department/colour
  (`GET /products`) or search by description (`GET /products/search`)
- A minimal browser page (`GET /app`) to try `/style` with a real photo and see
  the matched product images, instead of raw JSON

## Tech stack
- FastAPI, Python 3.11
- Gemini (Google AI) for photo analysis
- PostgreSQL (structured product data) via SQLAlchemy
- ChromaDB (semantic search) + Hugging Face sentence-transformers
- Docker Compose (api + db + chroma)

## Run with Docker (recommended)
```
cp .env.example .env  # add your Gemini API key
cd docker
docker compose up --build
```
The API is then available at http://localhost:8000 (try http://localhost:8000/app
for the browser page). Load the clothing database (H&M dataset via Hugging Face,
no Kaggle account needed):
```
docker compose exec api python -m scripts.ingest_hm_dataset --limit 5000
```
Drop `--limit` (or pass `--limit 0`) to ingest the full ~106k-item catalogue.

## Run locally without Docker
```
git clone https://github.com/prashastisahu/AI-LLM.git
cd AI-LLM
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # add your Gemini API key; DATABASE_URL defaults to a local SQLite file
uvicorn app.main:app --reload
```
Semantic search and product-matched `/style` results need a running Chroma
server (`docker run -p 8001:8000 chromadb/chroma:0.6.3` — version pinned to
match `requirements.txt`) — plain filtering (`GET /products`) and `/style`'s
text suggestions work without it.

## Tests
```
python -m pytest tests/ -v
```
