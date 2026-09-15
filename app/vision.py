import io
import json

from google import genai
from google.genai import types
from PIL import Image

from app.config import get_settings

settings = get_settings()
_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def prepare_image(image_bytes: bytes) -> bytes:
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = pil_image.convert("RGB")
    pil_image.thumbnail((1024, 1024), Image.LANCZOS)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


def _call_gemini(jpeg_bytes: bytes, prompt: str) -> dict:
    response = get_client().models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=prompt),
        ],
    )

    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]

    return json.loads(raw_text)


def build_analysis_prompt(user_text: str) -> str:
    return f"""You are a professional fashion stylist.

A user has uploaded a photo and written this request:
"{user_text}"

Look at the outfit carefully — its colours, style, mood, and formality. Identify 3-5
categories of clothing or accessories that would complement or elevate this look for
the occasion described. Do not suggest makeup.

For each category, write a short, descriptive search phrase (colours, style, material)
that could be used to search a real clothing catalogue for a matching item — describe
what you're looking for, not a specific invented product name.

Respond ONLY with valid JSON (no markdown, no commentary), matching exactly this schema:
{{
  "look_title": "3-5 word evocative title for this look",
  "look_description": "1-2 sentences describing the outfit and its vibe",
  "needs": [
    {{"category": "short category, e.g. Earrings, Bag, Outerwear, Shoes, Scarf", "search_query": "descriptive search phrase"}}
  ]
}}

Provide exactly 3-5 needs."""


def analyse_look(image_bytes: bytes, user_text: str) -> dict:
    """First RAG stage: understand the photo and decide what to search for."""
    jpeg_bytes = prepare_image(image_bytes)
    return _call_gemini(jpeg_bytes, build_analysis_prompt(user_text))


def build_synthesis_prompt(user_text: str, look: dict, candidates_by_category: dict[str, list[dict]]) -> str:
    candidates_block = ""
    for category, candidates in candidates_by_category.items():
        candidates_block += f"\nCATEGORY: {category}\nCANDIDATES:\n"
        if candidates:
            for c in candidates:
                candidates_block += (
                    f"- article_id: {c['article_id']} | {c['prod_name']} | "
                    f"{c['colour_group_name']} | {c['detail_desc']}\n"
                )
        else:
            candidates_block += "- (none found in the catalogue)\n"

    return f"""You are a professional fashion stylist finishing your recommendations.

Look: "{look.get('look_description', '')}"
The user asked: "{user_text}"

Below are real candidate products retrieved from the catalogue for each category you
identified. For each category, pick the single best candidate by its exact article_id
if one genuinely suits the look — or use null if none of the candidates are a good fit
(do not force a bad match). Then explain how to style your choice and why it works,
referencing its real details. If you chose null, give a general suggestion instead.
{candidates_block}
Respond ONLY with valid JSON (no markdown, no commentary), matching exactly this schema:
{{
  "recommendations": [
    {{"category": "...", "chosen_article_id": "article_id or null", "detail": "1-2 sentences", "reasoning": "1 sentence"}}
  ],
  "stylist_note": "one encouraging closing sentence, first person"
}}"""


def synthesise_recommendations(
    image_bytes: bytes, user_text: str, look: dict, candidates_by_category: dict[str, list[dict]]
) -> dict:
    """Second RAG stage: generate advice grounded in the retrieved real candidates."""
    jpeg_bytes = prepare_image(image_bytes)
    prompt = build_synthesis_prompt(user_text, look, candidates_by_category)
    return _call_gemini(jpeg_bytes, prompt)
