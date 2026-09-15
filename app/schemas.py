from pydantic import BaseModel, ConfigDict


class ClothingItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    article_id: str
    product_code: str
    prod_name: str
    product_type_name: str
    product_group_name: str
    colour_group_name: str
    department_name: str
    index_name: str
    section_name: str
    garment_group_name: str
    detail_desc: str
    image_url: str


class StyleRecommendation(BaseModel):
    category: str
    detail: str
    reasoning: str
    product: ClothingItemOut | None = None


class StyleResponse(BaseModel):
    look_title: str
    look_description: str
    recommendations: list[StyleRecommendation]
    stylist_note: str
    user_prompt: str
