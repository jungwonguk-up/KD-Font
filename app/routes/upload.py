# 업로드 라우터 
from pathlib import Path

from fastapi import APIRouter, Body, UploadFile, File, Form
# from pydantic_models.upload import UploadData

from library.func import create_workspace, read_image, save_image


upload_router = APIRouter(tags=["upload"])


# upload router (post)
# save img to local (not DB)
@upload_router.post("/")
async def upload_file(uuid: str = Form(None), file: UploadFile = File(...)) -> dict:
    print(uuid)

    # create workspace
    workspace = create_workspace(uuid)

    image = read_image(file)

    #filename
    file_path = Path(file.filename)
    # image full path
    img_full_path = workspace / file_path

    # image = Image.Image()
    save_image(image, str(img_full_path))
    
    return {"message": "image upload sucessfully"}