import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import os
from PIL import Image
import torchvision.models as models


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return identity + out
    
class Generator(nn.Module):
    def __init__(self, in_channels=3, num_residual_blocks=2):  # 进一步减少残差块的数量
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=9, stride=1, padding=4)  # 进一步减少通道数
        self.prelu = nn.PReLU()
        self.residual_blocks = self.make_layers(ResidualBlock, 8, num_residual_blocks)  # 进一步减少通道数
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)  # 进一步减少通道数
        self.bn2 = nn.BatchNorm2d(8)  # 进一步减少通道数
        self.upsample = self.make_layers(self.upsample_block, 8, 2)  # 进一步减少通道数
        self.conv3 = nn.Conv2d(8, in_channels, kernel_size=9, stride=1, padding=4)  # 进一步减少通道数

    # 其他代码保持不变

    def make_layers(self, block, in_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)
    
    def upsample_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out = self.residual_blocks(out1)
        out = self.bn2(self.conv2(out))
        out = out1 + out
        out = self.upsample(out)
        out = self.conv3(out)
        print(out.shape)
        return out

class SRDataset(Dataset):
    def __init__(self, low_res_img_path, transform=None):
        self.low_res_img_path = low_res_img_path
        self.low_res_images = os.listdir(low_res_img_path)
        self.transform = transform
        self.transformh = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        low_res_img = Image.open(low_res_img_path).convert("RGB")
        
        if self.transform:
            low_res_img = self.transform(low_res_img)
        
        return low_res_img

# 图像转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 初始化模型、损失和优化器
generator = Generator()
content_loss = ContentLoss()

criterion_GAN = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)


def color_transfer(src, target):
    # convert the images from the RGB to L*ab color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # 分别计算源图像和目标图像每个通道的平均值和标准差
    l_mean_src, a_mean_src, b_mean_src = np.mean(src_lab, axis=(0, 1))
    l_mean_tar, a_mean_tar, b_mean_tar = np.mean(target_lab, axis=(0, 1))
    l_std_src, a_std_src, b_std_src = np.std(src_lab, axis=(0, 1))
    l_std_tar, a_std_tar, b_std_tar = np.std(target_lab, axis=(0, 1))

    # 分别计算每个通道
    l, a, b = cv2.split(src_lab)
    l = ((l - l_mean_src) * (l_std_tar / l_std_src)) + l_mean_tar
    a = ((a - a_mean_src) * (a_std_tar / a_std_src)) + a_mean_tar
    b = ((b - b_mean_src) * (b_std_tar / b_std_src)) + b_mean_tar

    # 合并LAB通道
    transfer = cv2.merge([l, a, b])
    transfer = np.clip(transfer, 0, 255)
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    
    # return the color transferred image
    return transfer

generator.load_state_dict(torch.load(f"generator_epoch_100.pth"))

# 加载预训练模型
generator.eval()

# 测试集路径
test_low_res_dir = low_res_dir

# 创建测试数据集和数据加载器
test_dataset = SRDataset(test_low_res_dir,  transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 如果有GPU，使用GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
generator = generator.to(device)
content_loss = content_loss.to(device)



# 评估生成器
with torch.no_grad():
    test_time = 0
    for low_res in test_dataloader:
        low_res = low_res.to(device)
        gen_hr = generator(low_res)

        # gen_hr = F.interpolate(gen_hr, size=(high_res.size(2), high_res.size(3)), mode='bilinear', align_corners=False)

        # 将生成的高分辨率图像转换为PIL图像并保存
        gen_hr = gen_hr.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        low_res = low_res.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
            
        # 反归一化：假设原始数据范围为 [-1, 1]
        gen_hr = (gen_hr * 0.5 + 0.5) * 255
        gen_hr = gen_hr.clip(0, 255).astype('uint8')

        low_res = (low_res * 0.5 + 0.5) * 255
        low_res = low_res.clip(0, 255).astype('uint8')

        # gen_hr = color_transfer(gen_hr, low_res)
    
        gen_hr = Image.fromarray(gen_hr)
        gen_hr.save(f'output/gen_hr_2000{test_time}.png')



        test_time += 1
        if test_time == 1:
            break