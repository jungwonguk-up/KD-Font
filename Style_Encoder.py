import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import os

EPOCH = 100
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_availabe()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
run_name = "Style_Encoder"
print("Using Device:", DEVICE)
visualize = False

input_size = 64
batch_size = 8
device = "cuda"

# Set data directory
train_dirs = '/home/hojun/PycharmProjects/diffusion_font/code/make_font/Hangul_Characters_Image64'

# Set transform
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size,input_size)),
# #     torchvision.transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.ImageFolder(train_dirs,transform=transforms)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(64*64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)  # 입력의 특징을 3차원으로 압축합니다 (출력값이 바로 잠재변수가 됩니다.)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12), #디코더는 차원을 점차 64*64로 복원합니다.
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64*64),
            nn.Sigmoid(),       # 픽셀당 0과 1 사이로 값을 출력하는 sigmoid()함수를 추가합니다.
        )

    def forward(self, x):
        encoded = self.encoder(x) # encoder는 encoded라는 잠재변수를 만들고
        decoded = self.decoder(encoded) # decoder를 통해 decoded라는 복원이미지를 만듭니다.
        return encoded, decoded
    

autoencoder = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005) 
# Adam()을 최적화함수로 사용합니다. Adam은 SGD의 변형함수이며 학습중인 기울기를 참고하여 학습 속도를 자동으로 변화시킵니다.
criterion = nn.MSELoss() #원본값과 디코더에서 나온 값의 차이를 계산하기 위해 평균제곱오차(Mean Squared Loss) 오차함수를 사용합니다.

def train(autoencoder, train_loader):
    autoencoder.train()
    for step, (x, label) in enumerate(train_loader):
        x = x.view(-1, 64*64).to(DEVICE)
        y = x.view(-1, 64*64).to(DEVICE) #x(입력)와 y(대상 레이블)모두 원본이미지(x)인 것을 주의해야 합니다.
        label = label.to(DEVICE)

        encoded, decoded = autoencoder(x)

        loss = criterion(decoded, y) # decoded와 원본이미지(y) 사이의 평균제곱오차를 구합니다
        optimizer.zero_grad() #기울기에 대한 정보를 초기화합니다.
        loss.backward() # 기울기를 구합니다.
        optimizer.step() #최적화를 진행합니다.

#학습하기
model = autoencoder().to(device)
for epoch in range(1, EPOCH+1):
    train(model, train_loader)
    torch.save(model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))

if visualize == True:

    # 잠재변수를 3D 플롯으로 시각화
    view_data = trainset.data[:200].view(-1, 64*64) #원본이미지 200개를 준비합니다
    view_data = view_data.type(torch.FloatTensor)/255.
    test_x = view_data.to(DEVICE)
    encoded_data, _ = autoencoder(test_x)
    encoded_data = encoded_data.to("cpu")


    CLASSES = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

    fig = plt.figure(figsize=(10,8))
    ax = Axes3D(fig)

    X = encoded_data.data[:, 0].numpy()
    Y = encoded_data.data[:, 1].numpy()
    Z = encoded_data.data[:, 2].numpy() #잠재변수의 각 차원을 numpy행렬로 변환합니다.

    labels = trainset.targets[:200].numpy() #레이블도 넘파이행렬로 변환합니다.

    for x, y, z, s in zip(X, Y, Z, labels): #zip()은 같은 길이의 행렬들을 모아 순서대로 묶어줍니다.
        name = CLASSES[s]
        color = cm.rainbow(int(255*s/9))
        ax.text(x, y, z, name, backgroundcolor=color)

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()