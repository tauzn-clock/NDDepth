from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
from torchvision import transforms
import numpy as np
import torch
import os
import csv
import random

from dataloader.BaseDataloader import BaseImageData

class NYUImageData(BaseImageData):
    def __init__(self, root, *args, **kwargs):
        super(NYUImageData, self).__init__(root, *args, **kwargs)
        
        self.pixel_values = Image.open(os.path.join(root, args[0]))
        self.depth_values = Image.open(os.path.join(root, args[1]))
        self.normal_values = Image.open(os.path.join(root, args[2]))
        fx = float(args[3])
        fy = float(args[4])
        cx = float(args[5])
        cy = float(args[6])

        self.camera_intrinsics = np.array([[fx, 0, cx, 0],
                                           [0, fy, cy, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.camera_intrinsics = np.linalg.inv(self.camera_intrinsics)

        self.camera_intrinsics_resized = np.array([[fx/4, 0, cx/4, 0 ],
                                                   [0, fy/4, cy/4, 0],
                                                   [0, 0, 1, 0],
                                                   [0, 0, 0, 1]], dtype=np.float32)
        self.camera_intrinsics_resized = np.linalg.inv(self.camera_intrinsics_resized)

        self.depth_rescale = float(args[7])
        self.depth_max = float(args[8])

        H, W = self.pixel_values.size

        mask = np.zeros((W,H), dtype=np.bool)
        mask[45:471, 41:601] = True

        # Remove max and min values
        depth_np = np.array(self.depth_values)
        max_min_mask = (depth_np == depth_np.max()) | (depth_np == depth_np.min())
        mask = mask & ~max_min_mask

        self.mask = Image.fromarray(mask)
        
    def to_train(self):
        return preprocess_transform(train_transform(self))
    def to_test(self):
        return preprocess_transform(self)

def preprocess_transform(input):
    img_transform = transforms.Compose([
        #transforms.Resize((512, 512)),         # Resize images to 128x128
        #transforms.RandomRotation(30),         # Randomly rotate images between -30 and 30 degrees
        transforms.ToTensor(),                 # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    depth_transform = transforms.Compose([
        #transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    mask_transform = transforms.Compose([
        #transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    normal_transform = transforms.Compose([
        #transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    output = {}
    output["pixel_values"] = img_transform(input.pixel_values)
    output["depth_values"] = depth_transform(input.depth_values) / input.depth_rescale
    output["mask"] = mask_transform(input.mask)==1
    output["normal_values"] = normal_transform(input.normal_values)
    output["camera_intrinsics"] = torch.tensor(input.camera_intrinsics)
    output["camera_intrinsics_resized"] = torch.tensor(input.camera_intrinsics_resized)
    output["max_depth"] = input.depth_max
    
    return output

def train_transform(input):
    
    # Augment Brightness
    brightness = random.uniform(0.75, 1.25)
    enhancer = ImageEnhance.Brightness(input.pixel_values)
    input.pixel_values = enhancer.enhance(brightness)

    # Augment Color
    color = random.uniform(0.9, 1.1)
    enhancer = ImageEnhance.Color(input.pixel_values)
    input.pixel_values = enhancer.enhance(color)

    # Flip 
    if random.uniform(0,1) < 0.5:
        input.pixel_values = input.pixel_values.transpose(Image.FLIP_LEFT_RIGHT)
        input.depth_values = input.depth_values.transpose(Image.FLIP_LEFT_RIGHT)
        input.mask = input.mask.transpose(Image.FLIP_LEFT_RIGHT)
        input.normal_values = input.normal_values.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotate
    deg = random.uniform(-5,5)
    input.pixel_values = input.pixel_values.rotate(deg)
    input.depth_values = input.depth_values.rotate(deg)
    input.mask = input.mask.rotate(deg)
    input.normal_values = input.normal_values.rotate(deg)

    return input