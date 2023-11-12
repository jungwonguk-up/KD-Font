import os
from PIL import Image
import pandas as pd
from pyclovaocr import ClovaOCR
from fastapi import FastAPI

class OCRInspection:
    def __init__(self, image_dir, userid) -> None:
        self.image_dir = image_dir
        self.userid = userid
        self.label_list = []
        self.false_list = []
        self.ocr = ClovaOCR()
        self.Insepction_check = False

    def Inspection(self):
        data = {
            "userid": [],
            "label": [],
            "result": []
        }
        df = pd.DataFrame(data)
        self.Insepction_check = True
        for root, dirs, files in os.walk(self.image_dir):
            for dir in dirs:
                self.label_list.append(dir)  
                dir_path = os.path.join(root, dir)  
                for filename in os.listdir(dir_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(dir_path, filename)
                        result = self.ocr.run_ocr(
                            image_path=image_path,
                            language_code='ko',
                            ocr_mode='general'
                        )
                        if "text" in result and str(result["text"]) == str(dir_path[-1]): 
                            pass
                        else:
                            new_row = {
                                "userid": self.userid,
                                "label": dir_path[-1],
                                "result": "False"
                            }
                            df = df.append(new_row, ignore_index=True)
        json_file_path = f"{self.userid}_false.json"
        df.to_json(json_file_path, orient="records", lines=True)
        print(df)
        print("Inspection Done")
        return self.false_list

    def Couting(self):
        if self.Insepction_check == True:
            print(f"False Counint : {len(self.false_list)}")
            return len(self.false_list)
        else:
            print("Do Inspection First")

if __name__ == "__main__":
    # DB 접속 코드 작성
    app = FastAPI()
    @app.post("/process_images/{id}")
    async def inspection_images(userid: str, image_dir: str):
        inspection_processor = OCRInspection(image_dir, userid)
        inspection_processor.Inspection()
        return {"message": "Inspection_End"}

    userid = '123'
    image_dir = f'/home/wonguk/coding/OCR/imgs_{userid}/sample_stroke'  

    inspection_processor = OCRInspection(image_dir, userid)
    inspection_processor.Inspection()
    inspection_processor.Couting()

    # to do 파일 보내는 코드 작성

    # to do 리스트로 글자 묶어서 한번에 보내기
    