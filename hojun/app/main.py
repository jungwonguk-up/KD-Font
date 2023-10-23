from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database.db import Settings

import uvicorn


app = FastAPI()
settings = Settings()

app.include_router()

origins = ['*']
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


@app.on_event("startup")
async def init_db():
    await settings.initialize_db()


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)

