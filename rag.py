"""
RAG service: ChromaDB-backed product retrieval.
Products are embedded at ingest time; at query time we embed the
vision model's search_query and retrieve the closest matches.
"""
import json
import structlog
import chromadb
from chromadb.utils import embedding_functions

from app.core.config import get_settings
from app.models.schemas import Product, ProductCategory, Budget

logger = structlog.get_logger()
settings = get_settings()

# Budget price ceilings in EUR
BUDGET_CEILING: dict[str, float] = {
    "drugstore": 20.0,
    "mid": 80.0,
    "luxury": 99999.0,
    "any": 99999.0,
}


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_products(products: list[dict]) -> int:
    """
    Ingest scraped product dicts into ChromaDB.
    Call this after a scraper run. Returns number of products upserted.
    """
    collection = _get_collection()

    ids, documents, metadatas = [], [], []

    for p in products:
        doc = (
            f"{p['brand']} {p['name']} {p['category']} "
            f"{p.get('finish', '')} {p.get('description', '')} "
            f"shades: {' '.join(p.get('shades', []))}"
        )
        meta = {
            "id": p["id"],
            "name": p["name"],
            "brand": p["brand"],
            "category": p["category"],
            "price_eur": float(p.get("price_eur", 0)),
            "finish": p.get("finish", ""),
            "undertone_fit": p.get("undertone_fit", "all"),
            "skin_tones": json.dumps(p.get("skin_tones", [])),
            "occasions": json.dumps(p.get("occasions", [])),
            "url": p.get("url", ""),
            "image_url": p.get("image_url", ""),
            "source": p.get("source", ""),
            "shades": json.dumps(p.get("shades", [])),
        }
        ids.append(p["id"])
        documents.append(doc)
        metadatas.append(meta)

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("rag.ingest_complete", count=len(ids))
    return len(ids)


def search_products(
    query: str,
    category: str | None = None,
    undertone: str | None = None,
    budget: str = "any",
    n_results: int = 3,
) -> list[Product]:
    """
    Semantic search for products matching a natural-language query.
    Optionally filter by category, undertone fit, and budget ceiling.
    """
    collection = _get_collection()

    where_filters: list[dict] = []

    price_ceiling = BUDGET_CEILING.get(budget, 99999.0)
    where_filters.append({"price_eur": {"$lte": price_ceiling}})

    if category:
        where_filters.append({"category": {"$eq": category}})

    if undertone and undertone != "neutral":
        where_filters.append(
            {"undertone_fit": {"$in": [undertone, "all", "neutral"]}}
        )

    where = {"$and": where_filters} if len(where_filters) > 1 else where_filters[0]

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["metadatas", "distances"],
        )
    except Exception as e:
        logger.warning("rag.search_error", error=str(e), query=query)
        return []

    products = []
    for meta in (results["metadatas"] or [[]])[0]:
        try:
            products.append(
                Product(
                    id=meta["id"],
                    name=meta["name"],
                    brand=meta["brand"],
                    category=ProductCategory(meta["category"]),
                    price_eur=float(meta["price_eur"]),
                    shades=json.loads(meta.get("shades", "[]")),
                    finish=meta.get("finish") or None,
                    undertone_fit=meta.get("undertone_fit") or None,
                    skin_tones=json.loads(meta.get("skin_tones", "[]")),
                    occasions=json.loads(meta.get("occasions", "[]")),
                    url=meta.get("url", ""),
                    image_url=meta.get("image_url") or None,
                    source=meta.get("source", ""),
                )
            )
        except Exception as e:
            logger.warning("rag.product_parse_error", error=str(e))
            continue

    logger.info("rag.search", query=query[:60], hits=len(products))
    return products
