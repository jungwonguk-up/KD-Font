from odmantic import Model
from typing import Optional


class RequestModel(Model):
    uuid: str
    keyword: Optional[str]
    image: Optional[bytes]

    class config:
        collection = "queue"
