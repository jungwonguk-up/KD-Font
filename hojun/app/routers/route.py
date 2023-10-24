import base64
import json
import requests
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status
from beanie import PydanticObjectId

from database.db import Database
from models.basemodel import UserRequest
from library.func import get_storage_path, save_image, read_image

from typing import List


async def request_rest(id: str, image: bytes):
    inference_server_url = get_request("Inference_URL")
    headers = {"Content-Type": "application/json"}
    base64_image = base64.urlsafe_b64encode(image).decode("ascii")
    request_dict = {"inputs": {"id": id, "image": [base64_image]}}

    response = requests.post(
        inference_server_url,
        json.dumps(request_dict),
        headers=headers,
    )

    return dict(response.json())['outputs'][0]


user_router = APIRouter()
requests_database = Database(UserRequest)


@user_router.post("/")
async def create_inference_request(email: str = Form(...), image_file: UploadFile = File(...)) -> dict:
    # read image by pillow
    image_file = read_image(image_file)
    # make path and save image
    user_id = str(uuid.uuid4())
    storage_path = get_storage_path(user_id)
    image_path = storage_path / Path("ori.png") #TODO 확장자는 따로 지정해줘야하나?
    save_image(image_file, str(storage_path/"ori"))
    
    # new db
    user_request = UserRequest(
        id=user_id,
        email=email,
        original_image_path=str(image_path))
    
    print(f"id: {user_request.id}, email: {user_request.email}, ori_path: {user_request.original_image_path}")

    await requests_database.save(user_request)

    #TODO Request

    return {"message": "request info created sucessfully."}


@user_router.get("/{id}", response_model=UserRequest)
async def get_request(id: str) -> UserRequest:
    user_request = await requests_database.get(id)

    if not user_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="user_request info with ID dose not exist."
        )
    
    return user_request


@user_router.get("/", response_model=List[UserRequest])
async def get_all_requests() -> List[UserRequest]:
    user_requests = await requests_database.get_all()

    return user_requests


@user_router.delete("/{id}")
async def delete_request(id: str) -> dict:
    user_request = await requests_database.delete(id)
    
    if not user_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="user_requst with ID dose not exist."
        )
    
    return {"message": "Request info deleted successfully."}
