# 이미지 업로드 시 pydantic 모델
# DB 연결할지는 이후 판단 후 구현

from pydantic import BaseModel


class UploadData(BaseModel):
    img_path: str
    thumb_path: str

    #TODO: sample data