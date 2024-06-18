import numpy as np
from PIL import Image
import os
from torchvision.transforms import transforms

def load_image(image_path):
    image = Image.open(image_path)

    # 如果图像是RGBA，将透明背景转换为白色
    if image.mode == 'RGBA':
        # 创建一个白色背景
        background = Image.new('RGB', image.size, (255, 255, 255))
        # 合并图像，忽略透明度
        background.paste(image, mask=image.split()[3])  # 使用透明度通道作为蒙版
        image = background
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = transform(image)
        
    

    return image

import numpy as np
from PIL import Image

def save_image(image, path):
    image.save(path)