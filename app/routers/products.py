from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ClothingItem
from app.schemas import ClothingItemOut

router = APIRouter(prefix="/products", tags=["products"])


@router.get("", response_model=list[ClothingItemOut])
def list_products(
    category: str | None = None,
    department: str | None = None,
    colour: str | None = None,
    q: str | None = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[ClothingItem]:
    limit = min(limit, 100)
    stmt = select(ClothingItem)

    if category:
        stmt = stmt.where(ClothingItem.product_type_name.ilike(f"%{category}%"))
    if department:
        stmt = stmt.where(ClothingItem.department_name.ilike(f"%{department}%"))
    if colour:
        stmt = stmt.where(ClothingItem.colour_group_name.ilike(f"%{colour}%"))
    if q:
        stmt = stmt.where(
            ClothingItem.prod_name.ilike(f"%{q}%") | ClothingItem.detail_desc.ilike(f"%{q}%")
        )

    stmt = stmt.offset(offset).limit(limit)
    return list(db.execute(stmt).scalars().all())


@router.get("/search", response_model=list[ClothingItemOut])
def search_products(query: str, n_results: int = 10, db: Session = Depends(get_db)) -> list[ClothingItem]:
    from app.rag import search_items  # lazy import: keeps chromadb/sentence-transformers off the hot path for tests

    article_ids = search_items(query, n_results=n_results)
    if not article_ids:
        return []

    rows = {
        item.article_id: item
        for item in db.execute(
            select(ClothingItem).where(ClothingItem.article_id.in_(article_ids))
        ).scalars()
    }
    return [rows[aid] for aid in article_ids if aid in rows]


@router.get("/{article_id}", response_model=ClothingItemOut)
def get_product(article_id: str, db: Session = Depends(get_db)) -> ClothingItem:
    item = db.get(ClothingItem, article_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return item
