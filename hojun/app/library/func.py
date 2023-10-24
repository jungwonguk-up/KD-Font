import os.path
import uuid
from fastapi import UploadFile, HTTPException, status
from pathlib import Path
from PIL import Image

from library.get_config import get_config


def get_storage_path(id: str):
    """
    Return storage path. if not exsit, make new dir
    """
    # base directory
    storage_dir = Path(get_config("Image_storage_PATH"))
    # UUID to prevent file overwrite
    requset_id = Path(id)
    # path concat
    storage_path = storage_dir / requset_id

    # check exist, or create new workspace path
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    return storage_path


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
    
#TODO delete_image