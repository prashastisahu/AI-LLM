"""Ingest the H&M Personalized Fashion Recommendations dataset (via its Hugging
Face mirror, Qdrant/hm_ecommerce_products — CC BY 4.0, no Kaggle auth needed) into
Postgres (structured filtering) and Chroma (semantic search).

Usage:
    python -m scripts.ingest_hm_dataset --limit 5000
    python -m scripts.ingest_hm_dataset --limit 5000 --skip-chroma

Note: --limit uses streaming so only that many rows are downloaded. Omitting
--limit ingests the full ~106k catalogue, which downloads the whole dataset.
"""
import argparse
import itertools
import os

from datasets import load_dataset

from app.db import Base, SessionLocal, engine
from app.models import ClothingItem

DATASET_ID = "Qdrant/hm_ecommerce_products"

# The dataset's own `image_url` field points to a demo S3 bucket that's been
# taken down (404s on every key). This Hugging Face mirror hosts the actual H&M
# competition images and stays up as a normal HF dataset — build URLs from the
# article_id using its file layout instead of trusting the dataset's own field.
IMAGE_BASE_URL = "https://huggingface.co/datasets/einrafh/hnm-fashion-recommendations-data/resolve/main/data/raw/images"

FIELDS = [
    "article_id",
    "product_code",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "department_name",
    "index_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]


def row_to_item_kwargs(row: dict) -> dict:
    kwargs = {}
    for field in FIELDS:
        value = row.get(field)
        kwargs[field] = "" if value is None else str(value)

    article_id = kwargs["article_id"]
    kwargs["image_url"] = f"{IMAGE_BASE_URL}/{article_id[:3]}/{article_id}.jpg" if article_id else ""
    return kwargs


def ingest(limit: int | None, batch_size: int, skip_chroma: bool) -> None:
    Base.metadata.create_all(bind=engine)

    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    rows = itertools.islice(dataset, limit) if limit else dataset

    db = SessionLocal()
    batch: list[dict] = []
    total = 0

    try:
        for row in rows:
            batch.append(row_to_item_kwargs(row))
            if len(batch) >= batch_size:
                total += _flush_batch(db, batch, skip_chroma)
                batch = []
        if batch:
            total += _flush_batch(db, batch, skip_chroma)
    finally:
        db.close()

    print(f"Ingested {total} clothing items from {DATASET_ID}.", flush=True)


def _flush_batch(db, batch: list[dict], skip_chroma: bool) -> int:
    for kwargs in batch:
        db.merge(ClothingItem(**kwargs))
    db.commit()

    if not skip_chroma:
        from app.rag import get_collection, item_document

        collection = get_collection()
        collection.upsert(
            ids=[kwargs["article_id"] for kwargs in batch],
            documents=[item_document(kwargs) for kwargs in batch],
        )

    return len(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest the H&M clothing dataset")
    parser.add_argument("--limit", type=int, default=5000, help="Max rows to ingest (0 = full dataset)")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--skip-chroma", action="store_true", help="Only populate Postgres, skip embeddings")
    args = parser.parse_args()

    ingest(limit=args.limit or None, batch_size=args.batch_size, skip_chroma=args.skip_chroma)

    # torch/grpc leave background threads that crash Python's normal interpreter
    # shutdown (PyGILState_Release error) after all work is already done and
    # flushed above — force-exit cleanly instead of letting that teardown run.
    os._exit(0)
