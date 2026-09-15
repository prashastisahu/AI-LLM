"""Chroma-backed semantic search over clothing items.

Items are embedded at ingest time (see scripts/ingest_hm_dataset.py); at query
time we embed the free-text search query and retrieve the closest matches.
"""
import chromadb
from chromadb.utils import embedding_functions

from app.config import get_settings

settings = get_settings()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    _collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def item_document(item: dict) -> str:
    """Build the text that gets embedded for one clothing item."""
    return (
        f"{item['prod_name']} {item['product_type_name']} {item['product_group_name']} "
        f"{item['colour_group_name']} {item.get('detail_desc', '')}"
    )


def search_items(query: str, n_results: int = 10) -> list[str]:
    """Semantic search — returns matching article_ids ordered by similarity."""
    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=[],
    )
    return (results["ids"] or [[]])[0]
