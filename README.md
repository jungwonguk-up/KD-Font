# KoFont-Diffusion

**발표 영상**: 11/8 업로드 예정  
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white"/> <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=FastAPI&logoColor=white"/> <img src="https://img.shields.io/badge/TensorRT-FF6F00?style=flat&logo=TensorFlow&logoColor=white"/> 

# 프로젝트 한줄 소개

**나만의 손글씨 생성 서비스**는 사용자의 손글씨 **한 글자**를 입력받아 **diffusion model**을 이용하여 사용자의 손글씨 폰트를 만들어서 샘플 이미지를 보여주고 ttf파일로 손글씨 폰트를 보내주는 서비스 입니다. 
또한 해당 프로젝트는 CVPR, Neurips등의 논문 Accept를 목표로 모델 개발이 진행되고 있습니다.

# 팀원 소개
<table align="center">
    <tr height="160px">
        <td align="center" width="200px">
            <a href="https://github.com/jungwonguk"><img height="180px" width="180px" src="https://avatars.githubusercontent.com/u/98310175?v=4"/></a>
            <br />
            <a href="https://github.com/jungwonguk"><strong>정원국</strong></a>
        </td>
        <td align="center" width="200px">
            <a href="https://github.com/internationalwe"><img height="180px" width="180px" src=https://avatars.githubusercontent.com/u/46400961?v=4/></a>
            <br />
            <a href="https://github.com/internationalwe"><strong>신호준</strong></a>
        </td>
        <td align="center" width="200px">
            <a href="https://github.com/gih0109"><img height="180px" width="180px" src=https://avatars.githubusercontent.com/u/102566187?v=4/></a>
            <br />
            <a href="https://github.com/gih0109"><strong>송영섭</strong></a>
        </td>
    </tr>
    <tr height="40px">
        <td align="center" width="200px">
            <a href="https://guksblog.tistory.com/">Blog</a>
            <br/>
        </td>
        <td align="center" width="200px">
            <a></a> 
        </td>
        <td align="center" width="200px">
            <a href="https://dreamrunning.tistory.com/">Blog</a>
            <br/>
    </tr>
</table>

- 정원국
  - Team lead, Lightweight development, Model train & inference, Model architecture desgin, Development of various tools
- 신호준
  - Data Analysis & Converting, ML Pipline, Development of various tools, Model train & inference
- 송영섭
  - Backend Develop, Development of various tools, Model train & inference
---
# Project Version

### Version 1
- [x] Korean Model
- [x] Stroke Embedding Design
- [x] Self-Attention Model
- [x] ONNX, TensorRT Lightweight
- [x] Labeling Tool
- [x] DDPM, DDIM backbone, Stroke Type Design

### Version 1.5
- [x] OCR Model and Test Pipline
- [x] ML Pipline, Backend
- [x] Sample Image Generator
- [x] TTF Transformer
- [x] RLHF
- [x] UI/UX Desgin
- [x] Front End


### Version2
- [x] Cross-Attention Model
- [ ] Service Page
- [ ] Chinese Model
- [ ] Continuous Learning
- [ ] User Feedback Pipline
- [ ] Python Optimization

---

# 프로젝트 데모
### Font Generation Result
아래의 이미지와 같이 사용자의 손글씨에 맞는 **폰트를 생성**하고 이를 TTF 파일로 변환해줍니다.
![Screenshot from 2023-11-01 05-42-18](https://github.com/jungwonguk/KoFont-Diffusion/assets/98310175/70fe9b01-ec5b-429c-ac3a-4b459f3f08d7)

### Sample Image
아래의 이미지와 같이 변환된 폰트를 통해 시 문구를 **샘플이미지**로 보여줍니다.
<p align="center"><img src="https://github.com/jungwonguk/KoFont-Diffusion/assets/98310175/2dc6ea29-f81f-438b-abbc-8cfda906c58d"></p>

### **Web Demo**

11/8 업로드 예정

---
# Document
- Getting_started
  - [ML Train/Test](https://github.com/jungwonguk/KoFont-Diffusion/blob/main/ML/README.md)
  - [Light Weight](docs/PyTorch-Model-Convert.md)
- Demo
  - [How to Use ModelDeploy](docs/How-to-Use-ModelDeploy.md)





