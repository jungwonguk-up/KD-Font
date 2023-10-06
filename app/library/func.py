# 기타 함수들
import os.path
import uuid
import cv2
import numpy as np
from fastapi import UploadFile, HTTPException, status
from pathlib import Path
from PIL import Image

# import markdown
import functools
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    work_dir: str = "static/upload/"


settings = Settings()


def get_workspace(uuid):
    """
    Return workspace path
    """
    # base directory
    work_dir = Path(settings.work_dir)
    # UUID to prevent file overwrite
    requset_id = Path(uuid)
    # path concat
    workspace_path = work_dir / requset_id

    # check exist, or create new workspace path
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)

    return workspace_path


def read_image(file: UploadFile) -> Image:
    """return Pillow Image instance from uploadfile"""
    try:
        image = Image.open(file.file).convert("RGB")
    except:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail=f"{image.filename} is not image file."
        )
    return image


def save_image(image: Image, path: str, format="png"):
    """save image to path"""

    image.save(f"{path}.{format}", format=format)
    

