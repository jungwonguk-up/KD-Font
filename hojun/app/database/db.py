# mongoDB

from beanie import init_beanie, PydanticObjectId
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Any, List, Optional

from models.basemodel import User


# Setting Class
class Settings(BaseSettings):
    DATABASE_URL: Optional[str] = None # database url 

    async def initialize_db(self):
        client = AsyncIOMotorClient(self.DATABASE_URL)

        # init beanie
        await init_beanie(database=client.get_default_database(), document_models=User)

        class Config:
            env_file = '.env'


class Database:
    def __init__(self, model):
        self.model = model

    async def save(self, document) -> None:
        """Add New Recode to Database Collection"""
        await document.create()
        return
    
    async def get(self, id: PydanticObjectId) -> Any:
        """Get Recode by id"""
        doc = await self.model.get(id)
        if doc:
            return doc
        
        return False
    
    async def update(self, id: PydanticObjectId, body: BaseModel):
        """Update Recode"""
        doc_id = id
        des_body = body.model_dump()
        des_body = {k: v for k, v in des_body.items() if v is not None}
        update_query = {"$set": {field: value for field, value in des_body.items()}}

        doc = await self.get(doc_id)
        if not doc:
            return False

        await doc.update(update_query)
        return doc
    
    async def delete(self, id: PydanticObjectId) -> bool:
        """Delete Recode by id"""
        doc = await self.get(id)
        if not doc:
            return False
        
        await doc.delete()
        return True


