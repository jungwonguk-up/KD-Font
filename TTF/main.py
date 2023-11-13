from fastapi import FastAPI, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from png2ttf import PNGtoSVG, FontCreator, MakeTTF
from make_sample_img import make_example_from_ttf
from typing import List
import requests
import argparse
from pydantic import BaseModel
import json


class PathModel(BaseModel):
    id : str
    image : List[str]


parser = argparse.ArgumentParser()
parser.add_argument('--back_url', help=" : backend url")
parser.add_argument('--port', help=" : Set ttf port", default=8100 )

args = parser.parse_args()
backend_url = args.back_url

port = args.port

app = FastAPI()
# settings = Settings()
# requests_database = Database(UserRequest)

# 백엔드 주소
# 이미지 저장 주소
# 포트

@app.put("/")
async def push_ttf(data: PathModel) -> dict:
    # user_request = await requests_database.get(id)
    # sampled_img_list = user_request.sampling_images_path
    id = data.id
    path = data.image[0]   
    new_path = '/'.join(path.split('/')[:-2])
    MakeTTF(path,id)
    background_image_path = f'{new_path}/background/img.png'
    ttf_file_path = f'{new_path}/ttf/{id}_sample.ttf'
    text = '하늘을 우러러 \n 한 점 버그없기를 \n 자그만 디버깅에도 \n 나는 괴로워했다'
    make_example_from_ttf(text, background_image_path,ttf_file_path)

    response = requests.post(
        backend_url,
        data=json.dumps({"id" : id, "example_image_path" : f"{new_path}/example_img/example.jpg" })
    )

    return {"message" : "success" }

# @app.post("/")
# async def test_make_db(email:str = Form(...)) -> dict:
#     user_id ='123'
#     email = 'jdfidf@naver.com'
#     user_request = UserRequest(
#         id=user_id,
#         email=email,
#         sampling_images_path = ['/home/wonguk/coding/makettf/img/햱.png','/home/wonguk/coding/makettf/img/헂.png','/home/wonguk/coding/makettf/img/헔.png']
#     )
#     await requests_database.save(user_request)
#     return {"message" : "success" }


# @app.get("/{id}", response_model=UserRequest)
# async def get_request(id: str) -> UserRequest:
#     user_request = await requests_database.get(id)

#     if not user_request:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="user_request info with ID dose not exist."
#         )
    
#     return user_request


# @app.get("/", response_model=List[UserRequest])
# async def get_all_requests() -> List[UserRequest]:
#     user_requests = await requests_database.get_all()

#     return user_requests

# @app.delete("/{id}")
# async def delete_request(id: str) -> dict:
#     user_request = await requests_database.delete(id)
    
#     if not user_request:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="user_requst with ID dose not exist."
#         )
    
#     return {"message": "Request info deleted successfully."}



# @app.on_event("startup")
# async def init_db():
#     await settings.initialize_db()


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=int(port), reload=True)
