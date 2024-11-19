import sys
sys.path.append('/NDDepth/src')

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
import csv

from model import Model, ModelConfig
from dataloader import ImageDataset, preprocess_transform
from loss import silog_loss, rms_loss, get_metrics

torch.manual_seed(42)

train_dataset = ImageDataset('/scratchdata/nyu_data', '/scratchdata/nyu_data/data/nyu2_train.csv', transform=preprocess_transform)
train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, pin_memory=True)

test_dataset = ImageDataset('/scratchdata/nyu_data', '/scratchdata/nyu_data/data/nyu2_test.csv', transform=preprocess_transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, pin_memory=True)

csv_file = [["silog", "abs_rel", "log10", "rms", "sq_rel", "log_rms", "d1", "d2", "d3"]]
with open('metric.csv', mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    
    # Write all rows at once
    writer.writerows(csv_file)

config =  ModelConfig("micro07")
model = Model(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#if torch.cuda.device_count() > 1:
#    print(f"Training on {torch.cuda.device_count()} GPUs!")
#    model = nn.DataParallel(model)  # Wrap model for multi-GPU
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    model.train()
    running_loss = 0.0
    for i, x in enumerate(tqdm.tqdm(train_dataloader)):
        for k in x.keys():
            x[k] = x[k].to(device)
            
        optimizer.zero_grad()

        d1, d2 = model(x)
        
        gt = x["depth_values"]
        d1 = F.interpolate(d1[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
        d2 = F.interpolate(d2[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
                
        loss = silog_loss(d1, gt).mean() + rms_loss(d2, gt).mean()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        del x, gt, d1, d2
    
    print(f"Epoch {epoch} Loss: {running_loss / len(train_dataloader)}")
    torch.save(model, 'model.pth')
    
    model.eval()
    for i, x in enumerate(tqdm.tqdm(test_dataloader)):
        for k in x.keys():
            x[k] = x[k].to(device)
            
        d1, d2 = model(x)
        
        gt = x["depth_values"]
        d1 = F.interpolate(d1[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
        d2 = F.interpolate(d2[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
        d = (d1 + d2) /2
        
        metric = get_metrics(gt,d)
        
        """
        new_metric_save = []
        for m in metric:
            new_metric_save.append(m.item())
            print(m.item())
        print(new_metric_save)
        
        csv_file = []
        with open("metric.csv", mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                csv_file.append(row)  # Append each row to the data list
        csv_file.append(new_metric_save)
        with open("metric.csv", mode='a', newline='') as file:  # Open in append mode
            writer = csv.writer(file)
            writer.writerow(metric)  # Write the new row only
        """
    
        del x, gt, d1, d2, d
        