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
from app.schemas import ClothingItemOut  # noqa: E402

client = TestClient(app)

FAKE_GEMINI_RESPONSE = {
    "look_title": "Effortless Evening",
    "look_description": "A relaxed dark outfit with clean lines.",
    "recommendations": [
        {
            "category": "Earrings",
            "detail": "Minimalist gold hoops.",
            "reasoning": "Complements the dark, simple palette.",
            "search_query": "gold hoop earrings",
        },
        {
            "category": "Bag",
            "detail": "A structured crossbody bag.",
            "reasoning": "Keeps the look polished but casual.",
            "search_query": "crossbody bag",
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


def test_style_matches_real_product(monkeypatch):
    monkeypatch.setattr(style_router, "get_style_recommendation", lambda image_bytes, prompt: FAKE_GEMINI_RESPONSE)
    monkeypatch.setattr(
        style_router,
        "_find_matching_product",
        lambda search_query, db: ClothingItemOut.model_validate(db.get(ClothingItem, "2")),
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
    assert body["recommendations"][0]["category"] == "Earrings"
    assert body["recommendations"][0]["product"]["article_id"] == "2"


def test_style_gemini_failure_returns_502(monkeypatch):
    def raise_value_error(image_bytes, prompt):
        raise ValueError("invalid JSON")

    monkeypatch.setattr(style_router, "get_style_recommendation", raise_value_error)

    response = client.post(
        "/style",
        files={"image": ("outfit.jpg", _dummy_jpeg_bytes(), "image/jpeg")},
        data={"prompt": "evening dinner"},
    )

    assert response.status_code == 502
