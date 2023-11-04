# KD Font
Korean_Diffusion_Font 모델 Docs 입니다. 

## \# Stroke 설계
기존 Stroke Embedding 방식은 한국어 고유의 특징을 잘 반영하지 못하고 있습니다. 때문에 초성과 종성의 생김새와 특징을 구분해서 적용하지 못하고 있습니다. 이런 점을 개선하기 위해 새롭게 Stroke Embedding 방식을 설계하였습니다.
<p align="center"><img src="https://github.com/jungwonguk/KoFont-Diffusion/assets/98310175/fe7aa4cd-006f-4ddf-9430-9b9a54c35f9a"></p>




## \# 모델의 구조
모델의 기본 구조는 Diffusion입니다. Style, Strokes, Label을 받아들여 Embbedding Vector를 만들어냅니다.
- **V1**의 경우 **Self-Attention** 구조로 되어 있습니다.
- **V2**의 경우 **Cross-Attention** 구조로 되어 있어 초성,중성,종성의 **위치적 특징**을 더 잘 받아들입니다.
<p align="center"><img src="https://github.com/jungwonguk/KoFont-Diffusion/assets/98310175/d69d929a-4197-4003-a41e-36958bfcd1f3"></p>



## 개발 환경
```
Ubuntu 22.04
Nvidia RTX 4090
CUDA 
CUDNN 
pytorch
python 
```
## 훈련
```shell
# 패키지 설치

```

```shell
# 학습된 모델 다운 & 압축해제 & 덮어쓰기
.
├── font_python
│   ├── baseline
│   │   └── checkpoint
│   │       └── experiment_0_batch_16
│   │           ├── checkpoint
│   │           ├── unet.model-107850.data-00000-of-00001
│   │           ├── unet.model-107850.index
│   │           └── unet.model-107850.meta
 
```
[Download PreTrained Model](https://drive.google.com/file/d/1uLGAyY7zXUi2BHuc90-ILw-IgawVcsZ8/view?usp=sharing)

```shell
# font2image 데이터 생성
python 02_font2image.py 
```

```shell
# Train
python train.py
```

### Sampling_25Font
```shell
# package
python test.py
```

### Sampling_Total_Font
```shell
# package
python test.py
```





## \# Codes
```
common
├── dataset.py    # load dataset, data pre-processing
├── function.py   # deep learning functions : conv2d, relu etc.
├── models.py     # Generator(Encoder, Decoder), Discriminator
├── train.py      # model Trainer
└── utils.py      # data pre-processing etc.

get_data
├── font2img.py   # font.ttf -> image
└── package.py    # .png -> .pkl
```








　    


