"""
Vision service: sends image to Claude, extracts structured visual profile
and generates raw style recommendations before RAG enrichment.
"""
import base64
import json
import structlog
from anthropic import AsyncAnthropic

from app.core.config import get_settings
from app.models.schemas import StyleRequest, VisualProfile, ColorStory, ColorSwatch

logger = structlog.get_logger()
settings = get_settings()

SYSTEM_PROMPT = """You are MUSE, an expert AI stylist with deep knowledge of colour theory,
facial analysis, makeup artistry, and fashion. Analyse photos with the precision of a
professional colorist and makeup artist combined.

Your task: analyse the person's photo deeply — skin tone, undertone, eye colour, hair colour,
face shape, existing makeup/accessories — and return a structured JSON analysis.

Respond ONLY with valid JSON. No markdown, no backticks, no commentary."""

ANALYSIS_TEMPLATE = """Analyse this person's photo and return a JSON object with this exact schema:

{{
  "look_title": "3-5 word evocative aesthetic title",
  "look_description": "2-3 sentences on their natural features, colouring, and current vibe",
  "attributes": ["6-8 style keyword strings"],
  "visual_profile": {{
    "skin_tone": "e.g. medium-deep",
    "undertone": "warm | cool | neutral",
    "eye_color": "e.g. dark brown",
    "hair_color": "e.g. black with warm highlights",
    "face_shape": "e.g. oval",
    "features": ["notable feature 1", "notable feature 2"]
  }},
  "color_story": {{
    "palette": [
      {{"hex": "#RRGGBB", "name": "colour name", "use": "lips|eyes|blush|skin|accent"}}
    ],
    "description": "2 sentences on the colour direction and why it flatters"
  }},
  "raw_recommendations": [
    {{
      "category": "category name",
      "name": "specific product type + shade descriptor",
      "detail": "how to apply or wear it (2 sentences)",
      "reasoning": "why this suits their specific features (1 sentence)",
      "match_score": 85,
      "shade_suggestion": "specific shade name or descriptor",
      "search_query": "query to use when searching product database"
    }}
  ],
  "stylist_note": "One powerful, personal closing remark (20-30 words, first person)"
}}

Provide exactly 6 raw_recommendations covering: foundation/skin, eye makeup, lip colour,
blush or bronzer, jewellery or accessories, and one bonus (highlighter, eyeliner, nail colour, bag, etc.)

Context:
- Occasion: {occasion}
- Aesthetic: {aesthetic}
- Skin tone hint: {skin_tone_hint}
- Budget: {budget}

Be specific — real shade descriptors, finishes, formulas. The search_query field will be used
to retrieve matching products from a vector database."""


async def analyse_image(
    image_bytes: bytes,
    media_type: str,
    request: StyleRequest,
) -> dict:
    """
    Send image + preferences to Claude Vision and return structured analysis dict.
    The dict contains visual_profile, color_story, raw_recommendations, etc.
    """
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    prompt = ANALYSIS_TEMPLATE.format(
        occasion=request.occasion.value,
        aesthetic=request.aesthetic.value,
        skin_tone_hint=request.skin_tone_hint or "detect from image",
        budget=request.budget.value,
    )

    logger.info("vision.analyse_start", model=settings.vision_model)

    response = await client.messages.create(
        model=settings.vision_model,
        max_tokens=settings.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    try:
        result = json.loads(raw_text.strip())
    except json.JSONDecodeError as e:
        logger.error("vision.json_parse_error", error=str(e), raw=raw_text[:300])
        raise ValueError(f"Vision model returned invalid JSON: {e}")

    logger.info(
        "vision.analyse_complete",
        look_title=result.get("look_title"),
        recs=len(result.get("raw_recommendations", [])),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

    return result


def parse_visual_profile(data: dict) -> VisualProfile:
    vp = data.get("visual_profile", {})
    return VisualProfile(
        skin_tone=vp.get("skin_tone", "unknown"),
        undertone=vp.get("undertone", "neutral"),
        eye_color=vp.get("eye_color", "unknown"),
        hair_color=vp.get("hair_color", "unknown"),
        face_shape=vp.get("face_shape", "unknown"),
        features=vp.get("features", []),
    )


def parse_color_story(data: dict) -> ColorStory:
    cs = data.get("color_story", {})
    palette = [
        ColorSwatch(
            hex=s.get("hex", "#888888"),
            name=s.get("name", ""),
            use=s.get("use", ""),
        )
        for s in cs.get("palette", [])
    ]
    return ColorStory(palette=palette, description=cs.get("description", ""))
