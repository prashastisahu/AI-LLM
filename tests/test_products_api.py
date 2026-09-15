import os
import tempfile

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mktemp(suffix='.db')}"

from fastapi.testclient import TestClient  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models import ClothingItem  # noqa: E402

client = TestClient(app)

SAMPLE_ITEMS = [
    ClothingItem(
        article_id="1",
        product_code="C1",
        prod_name="Slim Fit Trousers",
        product_type_name="Trousers",
        product_group_name="Garment Lower body",
        colour_group_name="Black",
        department_name="Menswear",
        index_name="Menswear",
        section_name="Men Trend",
        garment_group_name="Trousers",
        detail_desc="Slim fit trousers in stretch cotton twill.",
        image_url="https://example.com/1.jpg",
    ),
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
    ),
]


def setup_module() -> None:
    db = SessionLocal()
    for item in SAMPLE_ITEMS:
        db.merge(item)
    db.commit()
    db.close()


def test_list_products_no_filter():
    response = client.get("/products")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_list_products_filter_by_category():
    response = client.get("/products", params={"category": "Coat"})
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert results[0]["article_id"] == "2"


def test_get_product_by_id():
    response = client.get("/products/1")
    assert response.status_code == 200
    assert response.json()["prod_name"] == "Slim Fit Trousers"


def test_get_product_not_found():
    response = client.get("/products/does-not-exist")
    assert response.status_code == 404
