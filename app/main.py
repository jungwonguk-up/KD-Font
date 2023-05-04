
from fastapi import FastAPI, Request, File, UploadFile
# from fastapi.responses import HTMLResponse, FileResponse
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



# # TODO : sampling 하기




@app.on_event("startup")
def on_app_start():
    """Before app start"""
    # TODO: connect DB
    # mongodb.connet()
    # TODO: load diffusion checkpoint
    # diffusion_model.load_state_dict(path=WEIGHT_PATH)


@app.on_event("shutdown")
def on_app_shutdown():
    """after app shutdown"""
    # mongodb.close()


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)