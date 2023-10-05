# diffusion 라우터

from fastapi import APIRouter, Body, UploadFile, File
import uuid

#TODO: 모델 작동 라우터
inference_router = APIRouter(tags=["inference"])


@inference_router.get("/")
async def inference(id):

    pass