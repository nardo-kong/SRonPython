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
    # Convert PIL Image to NumPy array if it's not already an array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Perform the multiplication and type conversion
    image = (image * 255).astype(np.uint8)  # Normalize to [0, 255]
    
    # Convert back to PIL Image and save
    image = Image.fromarray(image)
    image.save(path)