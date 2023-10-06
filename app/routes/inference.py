# inference 라우터
import os
import io

from fastapi import APIRouter, Body, UploadFile, File
from fastapi.responses import FileResponse
from pydantic_models.models import ImagePath
from library.func import get_workspace, read_image
import uuid

#TODO: 모델 작동 라우터
inference_router = APIRouter(tags=["inference"])


@inference_router.get("/{uuid}")
async def get_sample_image(uuid):

    path = get_workspace(uuid)
    print(path)
    sample_img_list = os.listdir(path)
    # print(sample_img_list)

    full_path = str(path) + '/file.png'
    
    return FileResponse(full_path, media_type="image/png")