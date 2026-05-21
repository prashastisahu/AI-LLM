# MUSE — AI Stylist

Multimodal makeup + accessory recommendation system.  
**Stack:** FastAPI · Claude Vision (Anthropic) · ChromaDB (RAG) · AWS (EC2 + S3) · Playwright (scraping)

---

## Architecture

```
User Photo + Prefs
       │
       ▼
  FastAPI /api/v1/style
       │
       ├─► S3: store image
       │
       ├─► Claude Vision (claude-opus-4-5)
       │     Multimodal analysis → VisualProfile + raw_recommendations
       │
       └─► RAG Pipeline (per recommendation)
             │
             ├─► ChromaDB semantic search
             │     (sentence-transformers embeddings)
             │
             └─► Merge: real product data + LLM reasoning
                         │
                         ▼
                   StyleResponse (JSON)
```

---

## Project Structure

```
ai-stylist/
├── app/
│   ├── api/routes.py          # FastAPI endpoints
│   ├── core/config.py         # Settings (pydantic-settings)
│   ├── models/schemas.py      # Pydantic models
│   └── services/
│       ├── vision.py          # Claude Vision analysis
│       ├── rag.py             # ChromaDB search + ingest
│       ├── s3.py              # AWS S3 image storage
│       └── stylist.py         # Orchestrator
├── scraper/
│   └── sephora.py             # Playwright scraper
├── scripts/
│   └── run_pipeline.py        # Scrape → ingest pipeline
├── tests/
│   └── test_stylist.py
├── infra/aws/
│   └── ec2_setup.sh           # EC2 deployment script
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/youruser/ai-stylist.git
cd ai-stylist
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys:
#   ANTHROPIC_API_KEY
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
#   DATABASE_URL (PostgreSQL on RDS)
#   REDIS_URL (ElastiCache)
```

### 3. Scrape + ingest products

```bash
# Scrape Sephora and ingest into ChromaDB in one command:
python scripts/run_pipeline.py --sources sephora \
    --categories foundation,blush,eyeshadow,lipstick,earrings \
    --max-per-category 100

# Or ingest from a pre-scraped JSON file:
python scripts/run_pipeline.py --from-file scraped_sephora.json
```

### 4. Run the API locally

```bash
uvicorn app.main:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

### 5. Test

```bash
pytest tests/ -v

# Manual test with curl:
curl -X POST http://localhost:8000/api/v1/style \
  -F "image=@your_photo.jpg" \
  -F "occasion=evening" \
  -F "aesthetic=bold" \
  -F "budget=mid"
```

---

## AWS Deployment

### EC2 (recommended for GPU/heavy workloads)

1. Launch an EC2 instance (t3.medium or larger, Ubuntu 22.04)
2. Copy your project: `rsync -av ./ ubuntu@your-ec2-ip:/opt/ai-stylist/`
3. Run the setup script: `bash infra/aws/ec2_setup.sh`
4. Your API will be live at `http://your-ec2-ip/api/v1/`

**AWS resources needed:**
- EC2: t3.medium (or t3.large if running scraper on same box)
- S3: one bucket for image storage (private, pre-signed URLs only)
- RDS: PostgreSQL (optional — for product metadata persistence)
- ElastiCache: Redis (optional — for request caching)

### Docker

```bash
docker build -t ai-stylist .
docker run -p 8000:8000 --env-file .env \
  -v $(pwd)/chroma_db:/app/chroma_db \
  ai-stylist
```

---

## API Reference

### `POST /api/v1/style`

**Form fields:**
| Field | Type | Default | Description |
|---|---|---|---|
| `image` | file | required | JPEG/PNG/WEBP, max 10MB |
| `occasion` | string | `everyday` | everyday / office / evening / special / outdoor |
| `aesthetic` | string | `natural` | natural / classic / bold / romantic / edgy |
| `skin_tone_hint` | string | null | fair / light / medium / olive / deep (null = auto-detect) |
| `budget` | string | `any` | any / drugstore / mid / luxury |

**Response:** `StyleResponse` JSON — look profile, visual analysis, colour palette, 6 recommendations.

### `GET /api/v1/health`
Health check.

### `POST /api/v1/products/ingest`
Admin: ingest a list of product dicts into ChromaDB. Protect with auth in production.

---

## Adding More Scrapers

The pipeline is designed to plug in new sources easily:

```python
# scraper/asos.py — follow the same pattern as sephora.py
# Then in scripts/run_pipeline.py:
if "asos" in sources:
    from scraper.asos import scrape_all as asos_scrape
    products.extend(await asos_scrape(categories, args.max_per_category))
```

---

## Notes

- The vision model used is `claude-opus-4-5` (best multimodal accuracy). Swap to `claude-sonnet-4-5` in `.env` for lower cost.
- ChromaDB uses `all-MiniLM-L6-v2` for embeddings (fast, runs on CPU). Upgrade to `all-mpnet-base-v2` for higher recall.
- Sephora's site structure changes — if selectors break, inspect updated HTML and update `scraper/sephora.py`.
- For production, add authentication to `/api/v1/products/ingest` and tighten CORS in `app/main.py`.
