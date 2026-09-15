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


def build_prompt(user_text: str) -> str:
    return f"""You are a professional fashion stylist.

A user has uploaded a photo and written this request:
"{user_text}"

Look at the outfit carefully — its colours, style, mood, and formality. Then suggest
clothing and accessory pieces that would complement or elevate the look, matching both
the outfit and what the user asked for. Do not suggest makeup.

Respond ONLY with valid JSON (no markdown, no commentary), matching exactly this schema:
{{
  "look_title": "3-5 word evocative title for this look",
  "look_description": "1-2 sentences describing the outfit and its vibe",
  "recommendations": [
    {{
      "category": "short category, e.g. Earrings, Bag, Outerwear, Shoes, Scarf",
      "detail": "1-2 sentences on how to style this piece with the outfit",
      "reasoning": "1 sentence on why this suits the photo and the user's request",
      "search_query": "a short text query to search a clothing catalogue for this item"
    }}
  ],
  "stylist_note": "one encouraging closing sentence, first person"
}}

Provide exactly 3-5 recommendations. Be specific — everything should clearly relate to
what you see in the image, not generic advice."""


def get_style_recommendation(image_bytes: bytes, prompt: str) -> dict:
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = pil_image.convert("RGB")
    max_size = (1024, 1024)
    pil_image.thumbnail(max_size, Image.LANCZOS)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    jpeg_bytes = buffer.read()

    response = get_client().models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=build_prompt(prompt)),
        ],
    )

    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]

    return json.loads(raw_text)
