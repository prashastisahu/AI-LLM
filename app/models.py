from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class ClothingItem(Base):
    __tablename__ = "clothing_items"

    article_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    product_code: Mapped[str] = mapped_column(String(20), index=True)
    prod_name: Mapped[str] = mapped_column(String(255))
    product_type_name: Mapped[str] = mapped_column(String(100), index=True)
    product_group_name: Mapped[str] = mapped_column(String(100), index=True)
    colour_group_name: Mapped[str] = mapped_column(String(100), index=True)
    department_name: Mapped[str] = mapped_column(String(100), index=True)
    index_name: Mapped[str] = mapped_column(String(100))
    section_name: Mapped[str] = mapped_column(String(100))
    garment_group_name: Mapped[str] = mapped_column(String(100))
    detail_desc: Mapped[str] = mapped_column(Text, default="")
    image_url: Mapped[str] = mapped_column(String(500), default="")
