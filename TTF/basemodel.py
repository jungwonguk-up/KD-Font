from beanie import Document
from pydantic import BaseModel, EmailStr
from typing import Optional, List


class UserRequest(Document):
    id: str
    email: EmailStr
    original_image_path: Optional[str] = None
    cropped_image_path: Optional[str] = None
    sampling_images_path: Optional[List[str]] = None
    example_image_path: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {"id": "1234 (uuid)",
                        "email": "abc@naver.com",
                        "original_image_path": ".../images/original_img.png",
                        "cropped_image_path": ".../images/cropped_img.png",
                        "sampling_images_path": [".../images/sample1.png", ".../images/sample2.png"],
                        "example_image_path": ".../images/example_img.png"}
        }
    }