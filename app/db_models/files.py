from odmantic import Model


class FileModel(Model):
    uuid: str
    keyword: str
    image: bytes

    class config:
        collection = "string"