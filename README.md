# KoFont-Diffusion

**발표 영상**: 11/8 업로드 예정  
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white"/> <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=FastAPI&logoColor=white"/> <img src="https://img.shields.io/badge/TensorRT-FF6F00?style=flat&logo=TensorFlow&logoColor=white"/> 

# 프로젝트 한줄 소개

**나만의 손글씨 생성 서비스**는 사용자의 손글씨 **한 글자**를 입력받아 **diffusion model**을 이용하여 사용자의 손글씨 폰트를 만들어서 샘플 이미지를 보여주고 ttf파일로 손글씨 폰트를 보내주는 서비스 입니다. 
또한 해당 프로젝트는 CVPR, Neurips등의 논문 등록을 목표로 모델 개발이 진행되고 있습니다.

# 팀원 소개
<table align="center">
    <tr height="160px">
        <td align="center" width="200px">
            <a href="https://github.com/jungwonguk"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/98310175?v=4"/></a>
            <br />
            <a href="https://github.com/jungwonguk"><strong>정원국</strong></a>
        </td>
        <td align="center" width="200px">
            <a href="https://github.com/jungwonguk"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/98310175?v=4"/></a>
            <br />
            <a href="https://github.com/jungwonguk"><strong>정원국</strong></a>
        </td>
        <td align="center" width="200px">
            <a href="https://github.com/jungwonguk"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/98310175?v=4"/></a>
            <br />
            <a href="https://github.com/jungwonguk"><strong>정원국</strong></a>
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
            <a href="https://jiyong-jeon.notion.site/Jeon-Jiyong-30ccaa36276d458ab0a8b1b06aab3c13">Notion</a>
            <br/>
    </tr>
</table>

- 정원국
  - Team lead, Lightweight, Model Train & Inference, Sampler Design, Test code
- 신호준
  - Data Analysis & Converting
- 송영섭
  - Backend
---
# Project Version

### Version 1
- [x] Self-Attention model
- [x] ONNX, TensorRT Lightweight
- [x] Labeling Tool
- [x] OCR model and test pipline
- [x] DDPM, DDIM backbone, Stroke Type Design
- [x] ML Pipline, Backend
- [x] Image Sampler
- [x] TTF Transformer


### Version2
- [x] Cross-Attention model
- [ ] Continuous Learning
- [ ] User Feedback Pipline
- [ ] Python Optimization

---

# 프로젝트 데모

### **Web Demo**

11/8 업로드 예정

---
# Document
- [프로젝트 소개](docs/introduce.md)
- Getting_started
  - [설치 방법](docs/Install.md)
  - [데이터 변환](KITTIVisualizer/Auto_transform.ipynb)
  - [PyTorch Model 변환](docs/PyTorch-Model-Convert.md)
- Demo
  - [How to Use ModelDeploy](docs/How-to-Use-ModelDeploy.md)
  - [How to build PyTorch for Jetson Xavier](docs/How-to-build-PyTorch-for-Jetson-Xavier.md)

---
# 폴더 구조
-
-
-
-

---
# 관련 파일

- 학습 모델 및 변환 모델 모음
  - 
