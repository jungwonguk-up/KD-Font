import base64
import json
import requests
import uuid
from pathlib import Path
import asyncio
import threading

from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from beanie import PydanticObjectId

from database.db import Database
from models.basemodel import UserRequest, CompleteSignal
from library.func import get_storage_path, save_image, read_image, get_filename
from library.img_process import image_processing, make_example_from_ttf
from library.get_config import get_config

from typing import List

import time


# 쓸 라우터랑 안쓸 라우터랑 분리하기?
# TODO: logging

INFERENCE_SERVER_URL = get_config("Inference_URL")
BACKEND_PORT = get_config("Port")
# EXAMPLE_TEXT = "괴그기깅나는늘다도디러로를만버없에우워을자점하한했"
EXAMPLE_TEXT = "가나다라"
EXAMPLE_BG_IMG = ""
# EXAMPLE_TEXT = get_config("example_text")
# EXAMPLE_BG_IMG = get_config("example_image_PATH")


def request_rest(id: str, cropped_img_path: str, text: str):
    inference_server_url = INFERENCE_SERVER_URL
    headers = {"Content-Type": "application/json"}

    request_dict = {"inputs": {"id": id, "cropped_img_path": cropped_img_path, "text": text, "port": BACKEND_PORT}}

    response = requests.post(
        inference_server_url,
        json.dumps(request_dict),
        headers=headers,
    )
    
    print(f"inference server response status code: {response.status_code}")


user_router = APIRouter()
requests_database = Database(UserRequest)


@user_router.post("/request")
async def create_inference_request(email: str = Form(...), image_file: UploadFile = File(...)) -> dict:

    # define user id by uuid
    user_id = str(uuid.uuid4())
    # read image by pillow
    image_file = await read_image(image_file)
    # get crop image after pre-processing
    cropped_image = image_processing(image_file, brightness_adj=1.5, contrast_enhance=2)
    # make path and save original & crop image
    ori_img_storage_path = get_storage_path(name="org")
    crop_img_storage_path = get_storage_path(name="crop")

    ori_path = str(ori_img_storage_path / Path(f"org_{user_id}.jpg"))
    crop_path = str(crop_img_storage_path / Path(f"crop_{user_id}.jpg"))

    await asyncio.gather(
        save_image(image_file, ori_path),
        save_image(cropped_image, crop_path)
    )

    # create new db recode 
    user_request = UserRequest(
        id=user_id,
        email=email,
        original_image_path=ori_path,
        cropped_image_path=crop_path
    )

    await requests_database.save(user_request)
    print(f"Request ID {user_id[:-8]} recoded to DB sucessfully.")

    threading.Thread(target=request_rest, 
                     args=(user_id, str(crop_path), EXAMPLE_TEXT)).start()
    
    # response = request_rest(id=user_id,
    #                               cropped_img_path=str(crop_path),
    #                               text=EXAMPLE_TEXT)
    
    return {"status": "success", "uuid": user_id}


@user_router.get("/status/{id}")
async def get_request_status(id: str):
    user_request = await requests_database.get(id)
    
    if not user_request:
        return {"status": "fail", "message": "user_request info with ID does not exist."}
        # raise HTTPException(
        #     status_code=status.HTTP_404_NOT_FOUND,
        #     detail="user_request info with ID dose not exist."
        # )

    
    if user_request.example_image_path is not None:
        return {"status": "success", "message": "Completed"}
    else:
        return {"status": "success", "message": "processing"}



@user_router.put("/request/{id}")
async def receive_sampling_complete_signal(data: CompleteSignal) -> dict:
    user_id = data.id
    body = {"example_image_path": data.example_image_path}
    user_request = await requests_database.update(id=user_id, body=body)

    if not user_request:
        return {"status": "fail"}

    return {"status": "success"}
    # return user_request


@user_router.get("/example_image/{id}")
async def get_example_image(id: str):
    user_request = await requests_database.get(id)

    if not user_request:
        return {"status": "fail", "message": "user_request info with ID dose not exist."}
        # raise HTTPException(
        #     status_code=status.HTTP_404_NOT_FOUND,
        #     detail="user_request info with ID dose not exist."
        # )
    # get example image path from db
    example_image_path = user_request.example_image_path

    if not example_image_path:
        return {"status": "fail", "message": "sample image path with ID dose not exist."}
        # raise HTTPException(
        #     status_code=status.HTTP_404_NOT_FOUND,
        #     detail="Sample image path with ID dose not exist."
        # )

    filename = get_filename(example_image_path)
    
    return FileResponse(path=example_image_path, filename=filename)


@user_router.put("/feedback/{id}")
async def feedback(id: str, feedback: str = Form(...)):
    body = {"user_feedback": feedback}
    user_request = await requests_database.update(id=id, body=body)

    if not user_request:
        return {"status": "fail"}

    return {"status": "success"}


@user_router.get("/request/{id}", response_model=UserRequest)
async def get_request(id: str) -> UserRequest:
    user_request = await requests_database.get(id)

    if not user_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="user_request info with ID dose not exist."
        )
    
    return user_request


@user_router.get("/get_all_db", response_model=List[UserRequest])
async def get_all_requests() -> List[UserRequest]:
    user_requests = await requests_database.get_all()

    return user_requests


# @user_router.put("/request/{id}", response_model=UserRequest)
# async def update_request(id: str, body: dict) -> UserRequest:
    
#     user_request = await requests_database.update(id=id, body=body)

#     if not user_request:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="user_request info with ID dose not exist."
#         )

#     return user_request


@user_router.delete("/request/{id}")
async def delete_request(id: str) -> dict:
    user_request = await requests_database.delete(id)
    
    if not user_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="user_requst with ID dose not exist."
        )
    
    return {"message": "Request info deleted successfully."}
