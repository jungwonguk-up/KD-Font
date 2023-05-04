from motor.motor_asyncio import AsyncIOMotorClient
from odmantic import AIOEngine
# from app.config import MONGO_DB_NAME, MONGO_URL


class MongoDB:
    def __init__(self) -> None:
        self.client = None
        self.engine = None

    def connet(self):
        self.client = AsyncIOMotorClient(MONGO_URL)
        self.engine = AIOEngine(client=self.client, database=MONGO_DB_NAME)
        print("Massege: DB와 성공적으로 연결이 되었습니다.")

    def close(self):
        self.client.close()


mongodb = MongoDB()