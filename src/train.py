import sys
sys.path.append('/NDDepth/src')

from PIL import Image
import torch
import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
from model import Model, ModelConfig
from dataloader import ImageDataset, preprocess_transform
from loss import silog_loss, histogram_intersection_loss, get_metrics

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = ImageDataset('/scratchdata/nyu_data', '/scratchdata/nyu_data/data/nyu2_train.csv', transform=preprocess_transform)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

test_dataset = ImageDataset('/scratchdata/nyu_data', '/scratchdata/nyu_data/data/nyu2_test.csv', transform=preprocess_transform)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

config =  ModelConfig("micro07")
model = Model(config).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    model.train()
    running_loss = 0.0

    cnt = 0
    for i, x in enumerate(tqdm.tqdm(train_dataloader)):
        cnt += 1
        for k in x.keys():
            x[k] = x[k].to(device)
            
        d1, d2 = model(x)
        
        gt = x["depth_values"]
        del x
        d1 = F.interpolate(d1[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
        d2 = F.interpolate(d2[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
                
        loss = silog_loss(d1, gt).mean() + histogram_intersection_loss(d2, gt).mean() * 10
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if cnt == 10:
            break
    
    print(f"Epoch {epoch} Loss: {running_loss / len(train_dataloader)}")
    
    model.eval()
    for i, x in enumerate(tqdm.tqdm(test_dataloader)):
        for k in x.keys():
            x[k] = x[k].to(device)
            
        d1, d2 = model(x)
        
        gt = x["depth_values"]
        del x
        d1 = F.interpolate(d1[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
        d2 = F.interpolate(d2[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
        d = (d1 + d2) /2
        
        print(get_metrics(gt, d))