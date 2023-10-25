import base64
import json
import requests
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from beanie import PydanticObjectId

from database.db import Database
from models.basemodel import UserRequest, UserRequestUpdate
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

    return None #TODO 전송 결과값을 받아야 한다


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
        original_image_path=str(image_path)
    )

    await requests_database.save(user_request)
    print(f"Request ID {user_id[:-8]} recoded to DB sucessfully.")

    #TODO Request
    # response = await request_rest(id=user_id,
    #                               image=image_file,)
    # if response:

    return {"message": "request info created sucessfully."}


@user_router.get("/{id}/status")
async def get_request_status(id: str):
    user_request = await requests_database.get(id)
    
    if not user_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="user_request info with ID dose not exist."
        )
    
    if user_request.cropped_image_path is None and user_request.sampling_images_path is None:
        return {"message": "Preprocessing"}
    elif user_request.cropped_image_path is not None and user_request.sampling_images_path is None:
        return {"message": "Sampling"}
    elif user_request.sampling_images_path is not None:
        return {"message": "Completed"}


#TODO # 샘플링 이미지 생성 완료 신호 받기
@user_router.post("/{id}")
async def receive_sampling_complete_signal(id: str) -> dict:
    user_request = await requests_database.get(id)
    
    sampling_images_path_list = user_request.sampling_images_path

    print(sampling_images_path_list)

    #TODO make example image and save to db

    return {"message": "signal received successfully."}


#TODO 샘플 이미지 받기
@user_router.get("/{id}/sample_image")
async def get_sampled_image(id: str):
    user_request = await requests_database.get(id)



    #TODO return multiple images
    return FileResponse()


@user_router.get("/{id}/example_image")
async def get_example_image(id: str):
    user_request = await requests_database.get(id)

    if not user_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="user_request info with ID dose not exist."
        )
    # get example image path from db
    example_image_path = user_request.example_image_path

    if not example_image_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample image path with ID dose not exist."
        )
    
    return FileResponse(path=example_image_path,)


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


#TODO should fix
@user_router.put("/{id}", response_model=UserRequest)
async def update_request(id: str, body: UserRequestUpdate) -> UserRequest:
    print("UserRequest")
    print(body)
    print()
    
    user_request = await requests_database.update(id=id, body=body)

    if not user_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="user_request info with ID dose not exist."
        )

    return user_request


@user_router.delete("/{id}")
async def delete_request(id: str) -> dict:
    user_request = await requests_database.delete(id)
    
    if not user_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="user_requst with ID dose not exist."
        )
    
    return {"message": "Request info deleted successfully."}
