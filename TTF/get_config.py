import yaml
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent


def get_config(key: str, 
               default_value: Optional[str] = None,
               yaml_path: str = str(BASE_DIR / "config.yaml")
               ):
    with open(yaml_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    try:
        return configs[key]
    except KeyError:
        if default_value:
            return default_value
        raise EnvironmentError(f"Set the {key} environment variable.")
    

# INF_SERVER_URL = get_config("Inference_URL")
# DATABASE_URL = get_config("Database_URL")
# STORAGE_PATH = get_config("Image_storage_PATH")
# print(INF_SERVER_URL)
# print(DATABASE_URL)
# print(STORAGE_PATH)