from fastapi import FastAPI
from routes.upload import upload_router
from routes.inference import inference_router

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware


import uvicorn


# define FastAPI instance
app = FastAPI()

#TODO: include_route
app.include_router(upload_router, prefix="/upload")
app.include_router(inference_router, prefix="/inference")


#TODO : middleware
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#TODO
# DB


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)