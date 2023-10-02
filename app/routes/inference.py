# diffusion 라우터

from fastapi import APIRouter, Body, UploadFile, File


#TODO: 모델 작동 라우터
inference_router = APIRouter(tags=["inference"])


@inference_router.get("/inference")
async def inference_diff():

    pass