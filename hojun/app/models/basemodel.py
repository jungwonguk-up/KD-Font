from beanie import Document
from pydantic import BaseModel, EmailStr
from typing import Optional, List


class User(Document):
    id: str
    email: EmailStr
    original_image: Optional[str]
    cropped_image: Optional[str]
    sampling_images: Optional[List[str]]
    example_image: Optional[str]

    model_config = {
        "json_schema_extra": {
            "example": {"id": "1234 (uuid)",
                        "email": "abc@naver.com",
                        "original_image": ".../images/original_img.png",
                        "cropped_image:": ".../images/cropped_img.png",
                        "sampling_images": [".../images/sample1.png", ".../images/sample2.png"],
                        "example_images": ".../images/example_img.png"}
        }
    }