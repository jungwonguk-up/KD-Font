# 업로드 라우터 
from pathlib import Path

from fastapi import APIRouter, Body, UploadFile, File


from library.func import create_workspace


upload_router = APIRouter(tags=["upload"])


# upload router (post)
# save img to local (not DB)
@upload_router.post("/new")
async def upload_file(file: UploadFile = File(...)) -> dict:

    # create workspace
    workspace = create_workspace()

    #filename
    file_path = Path(file.filename)
    # image full path
    img_full_path = workspace / file_path

    with open(str(img_full_path), 'wb') as myfile:
        contents = await file.read()
        myfile.write(contents)
    
    return {"message": "image upload sucessfully"}