from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import os
import csv

def preprocess_transform(output):
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
    
    output["pixel_values"] = img_transform(output["pixel_values"])
    output["depth_values"] = depth_transform(output["depth_values"])
    
    return output

class ImageData(object):
    def __init__(self, img, depth):
        super(ImageData, self).__init__()        
        self.intrinsic = None

        self.img_path = img
        self.depth_path = depth

class ImageDataset(Dataset):
    def __init__(self, root_dir, csv_path, transform = None):
        
        self.transform = transform
        
        self.dataset = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.dataset.append(ImageData(os.path.join(root_dir, row[0]), os.path.join(root_dir, row[1])))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        output = {}
        
        output["pixel_values"] = Image.open(self.dataset[idx].img_path)
        output["depth_values"] = Image.open(self.dataset[idx].depth_path)
    
        if self.transform:
            output = self.transform(output)
        return output

class ImageDataLoader(DataLoader):
    def __init__(self, root_dir, csv_path, batch_size, shuffle, num_workers, transform = None):
        self.dataset = ImageDataset(root_dir, csv_path, transform)
        super(ImageDataLoader, self).__init__(self.dataset, batch_size, shuffle, num_workers)