# 기타 함수들
import os.path
import uuid
from pathlib import Path
from PIL import Image
# import markdown
import functools
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    work_dir: str = "static/upload/"


settings = Settings()


def create_workspace():
    """
    Return workspace path
    """
    # base directory
    work_dir = Path(settings.work_dir)
    # UUID to prevent file overwrite
    requset_id = Path(str(uuid.uuid4())[:8])
    # path concat
    workspace_path = work_dir / requset_id

    # check exist, or create new workspace path
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)

    return workspace_path

