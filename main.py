import os
import base64
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from PIL import Image
import io

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="AI Stylist MVP")

# # ── Gemini model ──────────────────────────────────────────────────────────────
# model = genai.GenerativeModel("gemini-2.0-flash")

# ── Helper: build the prompt ──────────────────────────────────────────────────
def build_prompt(user_text:str) -> str:
    return f"""You are a professional fashion stylist and makeup artist.
 
A user has uploaded a photo and written this request:
"{user_text}"
 
Look at the outfit carefully — its colours, style, mood, and formality.
Then give personalised recommendations that match both the outfit and what the user said.
 
Respond in this exact format:
 
MAKEUP:
- Foundation: [finish and coverage suggestion]
- Eyes: [eye makeup suggestion]
- Lips: [lip colour and finish]
- Blush: [blush colour and placement]
 
ACCESSORIES:
- Earrings: [specific suggestion]
- Bag: [specific suggestion]
- One more: [any other accessory — bracelet, belt, scarf, shoes]
 
WHY THIS WORKS:
[2-3 sentences explaining why these suggestions complement this specific outfit and the user's request]
 
Be specific. Do not give generic advice. Everything should clearly relate to what you see in the image."""

# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/style")
async def get_style_recommendation(
    image: UploadFile =File(...),
    prompt: str = Form(...),
):
    
    # Read the upload image
    image_bytes = await image.read()

    # Open with Pillow to validate and resize if too large
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = pil_image.convert("RGB")
    # Resize if very large (keeps API call fast)
    max_size = (1024, 1024)
    pil_image.thumbnail(max_size, Image.LANCZOS)

    # Convert back to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    jpeg_bytes = buffer.read()

    # Send to Gemini - image + prompt together
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[
            types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=build_prompt(prompt)),
        ]
    )


    # Return the result
    return JSONResponse(content={
        "recommendatons": response.text,
        "user_prompt": prompt,
    })

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "AI Stylist MVP is running"}