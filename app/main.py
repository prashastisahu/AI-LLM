from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.db import Base, engine
from app.routers import products, style

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Stylist")

app.include_router(style.router)
app.include_router(products.router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/app")
def frontend() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/")
def root() -> dict:
    return {"status": "AI Stylist is running"}
