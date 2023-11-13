import os
import sys
import argparse
import yaml


def make_config(argv, args):

    config_data = {
        'Inference_URL': args.inf_url,
        'Database_URL': args.db_url,
        'Image_storage_PATH': args.s_path,
        'Port': args.port
    }

    with open("./config.yaml", "w") as f:
        yaml.dump(config_data, f)

parser = argparse.ArgumentParser()
parser.add_argument('--inf_url', help=' : Set inference server url')
parser.add_argument('--db_url', help=' : Set database server url')
parser.add_argument('--s_path', help=' : Set storage path')
parser.add_argument('--port', help=' : Set backend port', default=8000)

args = parser.parse_args()
argv = sys.argv

make_config(argv, args)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.route import user_router
from database.db import Settings

import uvicorn


app = FastAPI()
settings = Settings()

app.include_router(user_router)

origins = ["*"]
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


@app.on_event("startup")
async def init_db():
    await settings.initialize_db()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--inf_url', help=' : Set inference server url')
    # parser.add_argument('--db_url', help=' : Set database server url')
    # parser.add_argument('--s_path', help=' : Set storage path')
    # parser.add_argument('--port', help=' : Set backend port', default=8000)

    # args = parser.parse_args()
    # argv = sys.argv
    # make_config(argv, args)
    
    uvicorn.run("main:app", host="localhost", port=int(args.port), workers=5)
