from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn

# from app.models import mongodb


BASE_DIR = Path(__file__).resolve().parent # 절대경로


app = FastAPI()

templates = Jinja2Templates(directory= BASE_DIR / "templates") # derectory= : 템플릿에 사용할 html 파일 위치


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Font-Diffusion Inference"})



@app.get("/string", response_class=HTMLResponse)
async def search(request: Request, q: str):
    print(q)
    return templates.TemplateResponse(
        "./index.html", 
        {"request": request, "title": "Font-Diffusion Inference", "keyword": q}
        )


@app.on_event("startup")
def on_app_start():
    """Before app start"""
    # TODO: connect DB
    # mongodb.connet()
    # TODO: load diffusion checkpoint


@app.on_event("shutdown")
async def on_app_shutdown():
    """after app shutdown"""
    # mongodb.close()


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)