import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ClothingItem
from app.schemas import ClothingItemOut, StyleRecommendation, StyleResponse
from app.vision import analyse_look, synthesise_recommendations

logger = logging.getLogger(__name__)

router = APIRouter(tags=["style"])

CANDIDATES_PER_CATEGORY = 5


@router.post("/style", response_model=StyleResponse)
async def get_style(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    db: Session = Depends(get_db),
) -> StyleResponse:
    image_bytes = await image.read()

    try:
        look = analyse_look(image_bytes, prompt)
        candidates_by_category = {
            need.get("category", ""): _retrieve_candidates(need.get("search_query", ""), db)
            for need in look.get("needs", [])
        }
        result = synthesise_recommendations(image_bytes, prompt, look, candidates_by_category)
    except ValueError as e:
        raise HTTPException(status_code=502, detail="Gemini returned an unexpected response") from e

    recommendations = [
        StyleRecommendation(
            category=raw_rec.get("category", ""),
            detail=raw_rec.get("detail", ""),
            reasoning=raw_rec.get("reasoning", ""),
            product=_lookup_product(raw_rec.get("chosen_article_id"), db),
        )
        for raw_rec in result.get("recommendations", [])
    ]

    return StyleResponse(
        look_title=look.get("look_title", ""),
        look_description=look.get("look_description", ""),
        recommendations=recommendations,
        stylist_note=result.get("stylist_note", ""),
        user_prompt=prompt,
    )


def _retrieve_candidates(search_query: str, db: Session) -> list[dict]:
    if not search_query:
        return []

    try:
        from app.rag import search_items  # lazy: keeps chromadb off the hot path for tests

        article_ids = search_items(search_query, n_results=CANDIDATES_PER_CATEGORY)
    except Exception:
        logger.warning("Product retrieval failed for query %r", search_query, exc_info=True)
        return []

    if not article_ids:
        return []

    items = db.execute(select(ClothingItem).where(ClothingItem.article_id.in_(article_ids))).scalars()
    by_id = {item.article_id: item for item in items}
    return [
        {
            "article_id": aid,
            "prod_name": by_id[aid].prod_name,
            "colour_group_name": by_id[aid].colour_group_name,
            "detail_desc": by_id[aid].detail_desc,
        }
        for aid in article_ids
        if aid in by_id
    ]


def _lookup_product(article_id: str | None, db: Session) -> ClothingItemOut | None:
    if not article_id or article_id.lower() == "null":
        return None
    item = db.get(ClothingItem, article_id)
    return ClothingItemOut.model_validate(item) if item else None
