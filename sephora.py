"""
Sephora scraper — uses Playwright (headless Chromium) to scrape
product listings by category. Run via: python -m scraper.sephora

Output: list of product dicts compatible with rag.ingest_products()

Usage:
    python -m scraper.sephora --categories foundation,blush --max-per-category 100
"""
import asyncio
import json
import re
import uuid
import argparse
from pathlib import Path
from playwright.async_api import async_playwright
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

logger = structlog.get_logger()

SEPHORA_BASE = "https://www.sephora.com"

# Map our ProductCategory → Sephora URL path segment
CATEGORY_URLS: dict[str, str] = {
    "foundation": "/makeup/face-makeup/foundation",
    "concealer": "/makeup/face-makeup/concealer",
    "blush": "/makeup/face-makeup/blush",
    "bronzer": "/makeup/face-makeup/bronzer-contour",
    "highlighter": "/makeup/face-makeup/highlighter",
    "eyeshadow": "/makeup/eye-makeup/eyeshadow",
    "eyeliner": "/makeup/eye-makeup/eyeliner",
    "mascara": "/makeup/eye-makeup/mascara",
    "lipstick": "/makeup/lip-makeup/lipstick",
    "lip_gloss": "/makeup/lip-makeup/lip-gloss",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
async def scrape_category(
    page,
    category: str,
    url_path: str,
    max_products: int = 50,
) -> list[dict]:
    products = []
    url = f"{SEPHORA_BASE}{url_path}?pageSize=60"

    logger.info("sephora.scrape_start", category=category, url=url)

    await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    await page.wait_for_timeout(2000)  # let JS hydrate

    # Scroll to load lazy images
    for _ in range(3):
        await page.evaluate("window.scrollBy(0, window.innerHeight)")
        await page.wait_for_timeout(800)

    # Sephora product cards
    cards = await page.query_selector_all('[data-comp="ProductTile "]')
    logger.info("sephora.cards_found", count=len(cards))

    for card in cards[:max_products]:
        try:
            product = await _parse_card(card, category)
            if product:
                products.append(product)
        except Exception as e:
            logger.warning("sephora.card_parse_error", error=str(e))
            continue

    logger.info("sephora.scrape_done", category=category, count=len(products))
    return products


async def _parse_card(card, category: str) -> dict | None:
    # Product name
    name_el = await card.query_selector('[data-at="sku_item_name"]')
    brand_el = await card.query_selector('[data-at="sku_item_brand"]')
    price_el = await card.query_selector('[data-at="price"]')
    link_el = await card.query_selector("a")
    img_el = await card.query_selector("img")

    if not name_el or not brand_el:
        return None

    name = (await name_el.inner_text()).strip()
    brand = (await brand_el.inner_text()).strip()
    price_raw = (await price_el.inner_text()).strip() if price_el else "0"
    href = await link_el.get_attribute("href") if link_el else ""
    img_url = await img_el.get_attribute("src") if img_el else ""

    # Parse price — Sephora uses "$XX.XX" format, convert to EUR (approx)
    price_usd = _parse_price(price_raw)
    price_eur = round(price_usd * 0.92, 2)  # rough USD→EUR; use live rate in prod

    product_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"sephora:{href}"))

    return {
        "id": product_id,
        "name": name,
        "brand": brand,
        "category": category,
        "price_eur": price_eur,
        "url": f"{SEPHORA_BASE}{href}" if href.startswith("/") else href,
        "image_url": img_url,
        "source": "sephora",
        "shades": [],        # enriched in detail scrape pass
        "finish": None,      # enriched in detail scrape pass
        "description": f"{brand} {name}",
        "skin_tones": [],
        "undertone_fit": "all",
        "occasions": [],
    }


def _parse_price(raw: str) -> float:
    match = re.search(r"[\d,]+\.?\d*", raw.replace(",", ""))
    return float(match.group()) if match else 0.0


async def scrape_all(categories: list[str], max_per_category: int = 50) -> list[dict]:
    all_products = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        for cat in categories:
            url_path = CATEGORY_URLS.get(cat)
            if not url_path:
                logger.warning("sephora.unknown_category", category=cat)
                continue
            products = await scrape_category(page, cat, url_path, max_per_category)
            all_products.extend(products)
            await asyncio.sleep(2)  # polite delay between categories

        await browser.close()

    return all_products


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Sephora products")
    parser.add_argument(
        "--categories",
        default="foundation,blush,eyeshadow,lipstick",
        help="Comma-separated list of categories to scrape",
    )
    parser.add_argument("--max-per-category", type=int, default=50)
    parser.add_argument("--output", default="scraped_sephora.json")
    args = parser.parse_args()

    cats = [c.strip() for c in args.categories.split(",")]
    products = asyncio.run(scrape_all(cats, args.max_per_category))

    out_path = Path(args.output)
    out_path.write_text(json.dumps(products, indent=2))
    print(f"Saved {len(products)} products to {out_path}")
