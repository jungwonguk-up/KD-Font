# KD-Font Pipeline
Korean_Diffusion_Font 파이프라인 Docs 입니다. 

## \# Model Traing Pipeline
- 수집한 TTF 파일을 이용하여 각 TTF의 글자이미지 Dataset과 train.csv를 제작한 후 이를 Input Image로 활용하여 훈련 후 모델을 저장합니다.
<p align="center"><img src="https://github.com/jungwonguk/KD-Font/assets/46400961/c78c6e90-ee27-46b4-bee6-6c4824f4022e"></p>

## \# Continous Learning Pipeline
- 사용자에게 입력 받은 이미지를 생성 하여 Contionous Learning에 사용하고 재학습 한 후모델을 평가 한 후 서비스에 사용할 모델을 선정합니다.
<p align="center"><img src="https://github.com/jungwonguk/KD-Font/assets/46400961/35173bea-c6db-4329-bd4c-05d81454cc37"></p>


## \# Model Serving Pipeline
- 사용자가 입력한 이미지를 백엔드 서버에서 학습에 알맞게 자르고 이미지 향상 시킨 후 DB에 이미지 경로를 저장합니다.
- 인퍼런스 서버에서 처리한 이미지를 Sampling Image로 사용하여 시에 사용 할 25글자 이미지 생성 후 DB에 경로를 저장합니다.
- 생성한 이미지를 TTF로 생성 후 시이미지로 생성합니다.
- 이후 프론트엔드단에 생성한 시 이미지 경로를 전송합니다.
<p align="center"><img src="https://github.com/jungwonguk/KD-Font/assets/46400961/dd246b46-c97c-4fa0-b8bd-02d114642c97"></p>


## 개발 환경
```
Ubuntu 20.04
Nvidia RTX 4090
CUDA
CUDNN 
Pytorch
Python
Kubernetes
Kubeflow
Docker
Torchserve
Kserve
Fastapi
Mongodb

```
## 환경 설치

- MiniKube 설치
```bash

curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

- Kustomize 설치 

```bash
wget https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2Fv5.0.3/kustomize_v5.0.3_linux_amd64.tar.gz
tar -xvf kustomize_v5.0.3_linux_amd64.tar.gz
chmod +x kustomize
sudo mv kustomize /usr/local/bin/
```

- Kubectl 설치

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client --output=yaml
```


- Docker User에서 실행 가능하게 만들기

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

- MiniKube 실행

```bash
# docker driver option 설정
# kubernetes version v1.26.0 설정
# --extra-config 부분은 tokenRequest 활성화 관련 설정
# 실행 후 "Enabled addons: storage-provisioner, default-storageclass"가 뜨는 지 확인
minikube start --driver=docker \
--cpus='4' --memory='16g' \
--kubernetes-version=v1.26.0 \
--extra-config=apiserver.service-account-signing-key-file=/var/lib/minikube/certs/sa.key \
--extra-config=apiserver.service-account-issuer=kubernetes.default.svc
```

- Kubeflow 설치

```bash
# kubeflow manifests github 다운
git clone https://github.com/kubeflow/manifests
cd manifests
git checkout v1.7-branch
```

## Contionous Pipline 실행


### Pipeline Yaml 만들기
```shell
# First Learning Pipeline Yaml 생성
python ./kubeflow/KD_Font_kubeflow_first_training.py

# Continuous Learning & Serving Pipeline Yaml 생성
python ./kubeflow/KD_Font_kubeflow_continuous_training.py

# Serving Pipeline Yaml 생성
python ./kubeflow/KD_Font_kubeflow_servingy.py
```

### Pipeline 실행
- Kubeflow Central Dashboard 접속
```bash
# 실행 후 컴퓨터로컬ip:8080으로 접속
kubectl port-forward --address="컴퓨터로컬ip" svc/istio-ingressgateway -n istio-system 8080:80
```
- Central Dashborad 접속하여 Pipelines 탭에 진입하여 생성한 Yaml 파일을 Upload Pipeline 기능을 활용하여 Pipeline 생성
### Serving 실행
```shell
# torchserve 폴더 진입
cd torchserve
# mar 파일 생성
torch-model-archiver --model-name diffusion_torchserve --version 1.0 --serialized-file "weight 파일 이름" --handler "handler.py" --extra-file "style_enc.pth,model.py,utils.py,config.yaml"
```

```shell
# torchserve 실행
torchserve --start --model-store . --models diffusion_torchserve.mar --ts-config config.properties
```


## \# Codes
```shell
# 학습된 모델 다운 & 압축해제 & 덮어쓰기
├── ML
│   ├── torchserve
│   │   ├── config.properties
│   │   ├── config.yaml
│   │   ├── handler.py
│   │   ├── model.py
│   │   └── utils.py
│   │ 
│   │── kubeflow
│   │   ├── KD_Font_kubeflow_continuous_training.py
│   │   ├── KD_Font_kubeflow_first_training.py
│   │   └── KD_Font_kubeflow_first_training.py
```






　    


