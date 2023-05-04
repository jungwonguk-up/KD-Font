
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, Response
# from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field
from uuid import UUID, uuid4

from pathlib import Path
from typing import Optional
from datetime import datetime

import uvicorn

from inference import diffusion_model
from db_models.form import RequestModel

from db_models import mongodb

import io
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent # 절대경로
WEIGHT_PATH = "C:/Users/gih54/Desktop/diffusion/ckpt_290.pt" # 임시로 하드코딩


app = FastAPI()

# templates = Jinja2Templates(directory= BASE_DIR / "templates") # derectory= : 템플릿에 사용할 html 파일 위치



@app.get("/") # ("/", response_class=HTMLResponse)
async def root(request: Request):
    return {"hello": "world"}
    # return templates.TemplateResponse("index.html", {"request": request, "title": "Font-Diffusion Inference"})


class RequestSampling(BaseModel):
    id_: UUID = Field(default_factory=uuid4)
    request_str: Optional[str] = None
    request_img: Optional[UploadFile] = None
    # requested_at: datetime = Field(default_factory=datetime.now)


# 실행 테스트 페이지
@app.get("/sampling/{char}")
async def test_sampling(char: str):
    if len(char) > 1:
        return {"massage": "한 단어만 입력하세요"}
    
    img = diffusion_model.manual_sampling(char=char)
    imgbytes = io.BytesIO()
    img.save(imgbytes, format='png')
    
    return Response(content=imgbytes.getvalue(), media_type='image/png')


@app.on_event("startup")
def on_app_start():
    """Before app start"""
    # TODO: connect DB
    # mongodb.connet()
    # load diffusion checkpoint
    diffusion_model.load_state_dict(path=WEIGHT_PATH)
    print("diffusion model loaded!")


@app.on_event("shutdown")
def on_app_shutdown():
    """after app shutdown"""
    # mongodb.close()


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)