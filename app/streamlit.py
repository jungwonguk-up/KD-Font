import streamlit as st
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import uuid
import io


# set url
url = 'http://localhost:8000'


st.title("hello world")

img_file = st.file_uploader("이미지를 업로드 하세요", type=["jpg", "jpeg", "png"])

if st.button("upload"):

    if img_file is not None:

        image = Image.open(img_file)
        st.image(image)

        uuid = str(uuid.uuid4())[:8]
        uuid_dict = {"uuid": uuid}
        # uuid = json.dumps(uuid)
        print(uuid_dict)
        files = {"file": img_file.getvalue()}

        post_response = requests.post(url=url + '/upload/',
                                 data=uuid_dict,
                                 files=files,)
        
        st.write("image uploaed successfully.")
        print(post_response)

        get_response = requests.get(url=url + f'/inference/{uuid}')
        
        get_image = Image.open(io.BytesIO(get_response.content))
        st.image(get_image)
        
        
        # encoder = MultipartEncoder(
        #     fields={"file": {"filename", img_file, "image"}}
        # )

        # response = requests.post(url=url,
        #                          data=encoder,
        #                          headers={"Content-Type": encoder.content_type},
        #                          )