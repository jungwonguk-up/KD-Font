
from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status
from beanie import PydanticObjectId
import uuid

from database.db import Database
from models.basemodel import UserRequest

from typing import List


user_router = APIRouter()
requests_database = Database(UserRequest)


@user_router.post("/")
async def create_inference_request(email: str = Form(...), image: UploadFile = File(...)) -> dict:

    #TODO create image path
    
    # new db
    user_request = UserRequest(id=str(uuid.uuid4()),
                email=email,
                original_image=None, #TODO
                )
    
    print(f"id: {user_request.id}, email: {user_request.email}")

    await requests_database.save(user_request)

    return {"message": "request info created sucessfully."}


@user_router.get("/{id}", response_model=UserRequest)
async def get_request(id: str) -> UserRequest:
    user_request = await requests_database.get(id)

    if not user_request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="user_requst info with ID dose not exist."
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
