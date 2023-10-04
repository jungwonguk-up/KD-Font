import streamlit as st
import requests


# set url
url = 'http://localhost:8000/upload/new'


st.title("hello world")

img_file = st.file_uploader("이미지를 업로드 하세요", type=["jpg", "jpeg", "png"])


if st.button("upload"):

    if img_file:
        files = {"file": img_file.getvalue()}
        # data = img_file
        response = requests.post(url=url, 
                                 files=files)