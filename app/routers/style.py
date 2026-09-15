import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ClothingItem
from app.schemas import ClothingItemOut, StyleRecommendation, StyleResponse
from app.vision import get_style_recommendation

logger = logging.getLogger(__name__)

router = APIRouter(tags=["style"])


@router.post("/style", response_model=StyleResponse)
async def get_style(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    db: Session = Depends(get_db),
) -> StyleResponse:
    image_bytes = await image.read()

    try:
        data = get_style_recommendation(image_bytes, prompt)
    except ValueError as e:
        raise HTTPException(status_code=502, detail="Gemini returned an unexpected response") from e

    recommendations = [
        StyleRecommendation(
            category=raw_rec.get("category", ""),
            detail=raw_rec.get("detail", ""),
            reasoning=raw_rec.get("reasoning", ""),
            product=_find_matching_product(raw_rec.get("search_query") or raw_rec.get("category", ""), db),
        )
        for raw_rec in data.get("recommendations", [])
    ]

    return StyleResponse(
        look_title=data.get("look_title", ""),
        look_description=data.get("look_description", ""),
        recommendations=recommendations,
        stylist_note=data.get("stylist_note", ""),
        user_prompt=prompt,
    )


def _find_matching_product(search_query: str, db: Session) -> ClothingItemOut | None:
    if not search_query:
        return None

    try:
        from app.rag import search_items  # lazy: keeps chromadb off the hot path for tests

        article_ids = search_items(search_query, n_results=1)
    except Exception:
        logger.warning("Product search failed for query %r", search_query, exc_info=True)
        return None

    if not article_ids:
        return None

    item = db.get(ClothingItem, article_ids[0])
    return ClothingItemOut.model_validate(item) if item else None
