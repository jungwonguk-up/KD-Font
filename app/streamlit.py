import streamlit as st
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import uuid


# set url
url = 'http://localhost:8000/upload/'


st.title("hello world")

img_file = st.file_uploader("이미지를 업로드 하세요", type=["jpg", "jpeg", "png"])

if st.button("upload"):

    if img_file is not None:

        image = Image.open(img_file)
        st.image(image)
        
        uuid = {"uuid": str(uuid.uuid4())[:8]}
        # uuid = json.dumps(uuid)
        print(uuid)
        files = {"file": img_file.getvalue()}

        response = requests.post(url=url,
                                 data=uuid,
                                 files=files,)
        
        st.write("image uploaed successfully.")
        print(response)
        
        
        # encoder = MultipartEncoder(
        #     fields={"file": {"filename", img_file, "image"}}
        # )

        # response = requests.post(url=url,
        #                          data=encoder,
        #                          headers={"Content-Type": encoder.content_type},
        #                          )