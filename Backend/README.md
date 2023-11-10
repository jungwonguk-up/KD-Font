## KoFont Diffusion Backend

KoFont 생성 서비스를 위한 백엔드 서버 및 서버 구축


## Requirement

#### Python
- python >= 3.10

#### Backend and MongDB
- fastapi >= 0.103.0
- beanie >= 1.22.6
- pydantic >= 2.4.2
- pydantic-settings >=2.0.3

#### ETC
- opencv, pillow, imutils, numpy


## Set-Up

1. Backend/app 폴더 내 `config.yaml` 파일을 생성 후 다음 내용을 작성합니다.

    ```
    Inference_URL: 인퍼런스 서버 주소
    Database_URL: MongoDB 주소
    Image_storage_PATH: 이미지 저장 Storage 경로 (외부 Storage 포함)
    TEXT: 예시로 생성할 글자 (예: 가나다라)
    ``` 

2. MongoDB 인스턴스를 시작합니다.

    ```
    (venv)$ mongod --dbpath DB경로
    ```

3. main.py 를 실행하여 FastAPI 서버를 실행합니다.

    ```
    (venv)$ python main.py
    ```