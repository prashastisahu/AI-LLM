# AI Stylist
![CI](https://github.com/prashastisahu/AI-LLM/actions/workflows/ci.yml/badge.svg)
An AI-powered fashion and makeup recommendation system. Upload an outfit 
photo and get personalised makeup and accessory suggestions powered by 
Gemini Vision API and a RAG pipeline with ChromaDB.

## What it does
- Analyses outfit photos using Gemini 2.0 Vision
- Retrieves matching products via semantic search (ChromaDB + Hugging Face 
  sentence-transformers)
- Returns structured makeup and accessory recommendations

## Tech stack
- FastAPI, Python 3.11
- Gemini 2.0 Flash (Google AI)
- ChromaDB (vector database)
- Hugging Face sentence-transformers (embeddings)
- PostgreSQL (coming soon)

## How to run locally
git clone https://github.com/prashastisahu/AI-LLM.git
cd AI-LLM
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # add your Gemini API key
uvicorn main:app --reload