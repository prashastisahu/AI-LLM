import io
import os
import tempfile

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mktemp(suffix='.db')}"

from PIL import Image  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.routers.style as style_router  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import ClothingItem  # noqa: E402

client = TestClient(app)

FAKE_LOOK = {
    "look_title": "Effortless Evening",
    "look_description": "A relaxed dark outfit with clean lines.",
    "needs": [
        {"category": "Earrings", "search_query": "gold hoop earrings"},
        {"category": "Bag", "search_query": "crossbody bag"},
    ],
}

FAKE_SYNTHESIS = {
    "recommendations": [
        {
            "category": "Earrings",
            "chosen_article_id": "2",
            "detail": "The wool coat's grey tone pairs well with warm gold.",
            "reasoning": "Complements the dark, simple palette.",
        },
        {
            "category": "Bag",
            "chosen_article_id": None,
            "detail": "A structured crossbody bag in any neutral tone.",
            "reasoning": "None of the retrieved candidates quite fit.",
        },
    ],
    "stylist_note": "You've got effortless style already — these just complete it.",
}


def setup_module() -> None:
    db = SessionLocal()
    db.merge(
        ClothingItem(
            article_id="2",
            product_code="C2",
            prod_name="Wool Coat",
            product_type_name="Coat",
            product_group_name="Garment Upper body",
            colour_group_name="Grey",
            department_name="Womenswear",
            index_name="Ladieswear",
            section_name="Women Outerwear",
            garment_group_name="Outdoor",
            detail_desc="Tailored wool-blend coat.",
            image_url="https://example.com/2.jpg",
        )
    )
    db.commit()
    db.close()


def _dummy_jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10), color=(0, 0, 0)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_style_retrieve_then_generate(monkeypatch):
    monkeypatch.setattr(style_router, "analyse_look", lambda image_bytes, prompt: FAKE_LOOK)
    monkeypatch.setattr(style_router, "_retrieve_candidates", lambda search_query, db: [])
    monkeypatch.setattr(
        style_router,
        "synthesise_recommendations",
        lambda image_bytes, prompt, look, candidates_by_category: FAKE_SYNTHESIS,
    )

    response = client.post(
        "/style",
        files={"image": ("outfit.jpg", _dummy_jpeg_bytes(), "image/jpeg")},
        data={"prompt": "evening dinner"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["look_title"] == "Effortless Evening"
    assert len(body["recommendations"]) == 2
    assert body["recommendations"][0]["product"]["article_id"] == "2"
    assert body["recommendations"][1]["product"] is None


def test_style_gemini_failure_returns_502(monkeypatch):
    def raise_value_error(image_bytes, prompt):
        raise ValueError("invalid JSON")

    monkeypatch.setattr(style_router, "analyse_look", raise_value_error)

    response = client.post(
        "/style",
        files={"image": ("outfit.jpg", _dummy_jpeg_bytes(), "image/jpeg")},
        data={"prompt": "evening dinner"},
    )

    assert response.status_code == 502
