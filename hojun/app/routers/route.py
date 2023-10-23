
from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status
from beanie import PydanticObjectId
import uuid

from database.db import Database
from models.basemodel import User

from typing import List


user_router = APIRouter()
user_database = Database(User)


@user_router.post("/upload")
async def create_inference_request(email: str = Form(...), image: UploadFile = File(...)) -> dict:

    #TODO create image path
    
    # new db
    user = User(id=str(uuid.uuid4()),
                email=email,
                original_image=None, #TODO
                )
    
    print(user.id, user.email)

    # await user_database.save(user)

    return {"message": "image uploaded sucessfully."}
