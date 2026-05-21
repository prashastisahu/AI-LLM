"""
Stylist orchestrator: ties together vision analysis → RAG retrieval → final response.

Flow:
  1. Upload image to S3
  2. Claude Vision analyses image → VisualProfile + raw_recommendations
  3. For each recommendation, search ChromaDB for matching real products
  4. Merge LLM reasoning with real product data → Recommendation objects
  5. Return complete StyleResponse
"""
import structlog

from app.models.schemas import (
    StyleRequest, StyleResponse, Recommendation, Product
)
from app.services import vision, rag, s3

logger = structlog.get_logger()


async def run_style_analysis(
    image_bytes: bytes,
    media_type: str,
    request: StyleRequest,
) -> StyleResponse:
    # 1. Upload to S3 (non-blocking — fire and use the key)
    s3_key = s3.upload_image(image_bytes, media_type)
    logger.info("stylist.image_uploaded", key=s3_key)

    # 2. Vision analysis
    vision_data = await vision.analyse_image(image_bytes, media_type, request)
    visual_profile = vision.parse_visual_profile(vision_data)
    color_story = vision.parse_color_story(vision_data)

    # 3. RAG enrichment — for each raw recommendation, search for real products
    recommendations: list[Recommendation] = []

    for raw_rec in vision_data.get("raw_recommendations", []):
        matched_product: Product | None = None

        search_query = raw_rec.get("search_query") or raw_rec.get("name", "")
        category_str = _normalise_category(raw_rec.get("category", ""))

        rag_hits = rag.search_products(
            query=search_query,
            category=category_str,
            undertone=visual_profile.undertone,
            budget=request.budget.value,
            n_results=3,
        )

        if rag_hits:
            # Pick best hit — RAG returns by cosine similarity, first = best
            matched_product = rag_hits[0]
            display_name = f"{matched_product.brand} {matched_product.name}"
        else:
            # No product in DB yet — use LLM-suggested name as fallback
            display_name = raw_rec.get("name", "")

        rec = Recommendation(
            category=raw_rec.get("category", ""),
            product=matched_product,
            name=display_name,
            detail=raw_rec.get("detail", ""),
            reasoning=raw_rec.get("reasoning", ""),
            match_score=int(raw_rec.get("match_score", 80)),
            shade_suggestion=raw_rec.get("shade_suggestion"),
        )
        recommendations.append(rec)

    response = StyleResponse(
        look_title=vision_data.get("look_title", "Your Look"),
        look_description=vision_data.get("look_description", ""),
        attributes=vision_data.get("attributes", []),
        visual_profile=visual_profile,
        color_story=color_story,
        recommendations=recommendations,
        stylist_note=vision_data.get("stylist_note", ""),
        image_s3_key=s3_key,
    )

    logger.info(
        "stylist.complete",
        look_title=response.look_title,
        recs=len(response.recommendations),
        rag_hits=sum(1 for r in response.recommendations if r.product),
    )
    return response


def _normalise_category(raw: str) -> str | None:
    """Map free-form category strings from the LLM to ProductCategory enum values."""
    raw = raw.lower().strip()
    mapping = {
        "foundation": "foundation",
        "concealer": "concealer",
        "blush": "blush",
        "bronzer": "bronzer",
        "highlighter": "highlighter",
        "eyeshadow": "eyeshadow",
        "eye shadow": "eyeshadow",
        "eyeliner": "eyeliner",
        "eye liner": "eyeliner",
        "mascara": "mascara",
        "lipstick": "lipstick",
        "lip colour": "lipstick",
        "lip color": "lipstick",
        "lip gloss": "lip_gloss",
        "earrings": "earrings",
        "necklace": "necklace",
        "bracelet": "bracelet",
        "bag": "bag",
        "nail": "nail_colour",
        "hair": "hair_accessory",
    }
    for key, val in mapping.items():
        if key in raw:
            return val
    return None  # no filter — let RAG search across all categories
