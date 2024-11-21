from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import os
import csv

class BaseImageData(object):
    def __init__(self, root, *args, **kwargs):
        super(BaseImageData, self).__init__()      
        self.camera_intrinsics = None
        self.pixel_values = None
        self.depth_values = None
        self.normal_values = None
        self.mask = None
    def to_train(self):
        return 
    def to_test(self):
        return
    
class BaseImageDataset(Dataset):
    def __init__(self, dataset_type, data_class, root_dir, csv_path):
        self.dataset_type = dataset_type
        self.data_class = data_class
        self.root_dir = root_dir
        
        self.dataset = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.dataset.append(row)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset_type == "train":
            return self.data_class(self.root_dir, *self.dataset[idx]).to_train()
        elif self.dataset_type == "test":
            return self.data_class(self.root_dir, *self.dataset[idx]).to_test()
        else:
            raise ValueError("Invalid dataset type")