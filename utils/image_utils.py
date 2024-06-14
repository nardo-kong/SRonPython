import numpy as np
from PIL import Image
import os
from torchvision.transforms import transforms

def load_image(image_path):
    image = Image.open(image_path)
    

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