import sys
sys.path.append('/NDDepth/src')

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
import csv
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from model import Model, ModelConfig
from dataloader import ImageDataset, preprocess_transform
from loss import silog_loss, rms_loss, get_metrics

torch.manual_seed(42)

def init_process_group(local_rank, world_size):
    dist.init_process_group(
        backend='nccl',  # Use NCCL backend for multi-GPU communication
        rank=local_rank,
        world_size=world_size
    )
    torch.cuda.set_device(local_rank)  # Set the GPU device for the current process

def main(local_rank, world_size):
    init_process_group(local_rank, world_size)

    train_dataset = ImageDataset('/scratchdata/nyu_data', '/scratchdata/nyu_data/data/nyu2_train.csv', transform=preprocess_transform)
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    train_dataloader = DataLoader(train_dataset, batch_size=6, pin_memory=True, sampler=train_sampler)

    test_dataset = ImageDataset('/scratchdata/nyu_data', '/scratchdata/nyu_data/data/nyu2_test.csv', transform=preprocess_transform)
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank)
    test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

    csv_file = [["silog", "abs_rel", "log10", "rms", "sq_rel", "log_rms", "d1", "d2", "d3"]]
    with open('metric.csv', mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        
        # Write all rows at once
        writer.writerows(csv_file)

    config =  ModelConfig("tiny07")
    model = Model(config).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for i, x in enumerate(tqdm.tqdm(train_dataloader)):
            for k in x.keys():
                x[k] = x[k].to(local_rank)
                            
            optimizer.zero_grad()

            d1, d2 = model(x)
            
            gt = x["depth_values"]
            d1 = F.interpolate(d1[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
            d2 = F.interpolate(d2[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
                    
            loss = silog_loss(d1, gt).sum() + rms_loss(d2, gt).sum()
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
                    
        print(f"Epoch {epoch} Loss: {running_loss / len(train_dataloader)}")
        torch.save(model, 'model.pth')
        
        model.eval()
        tot_metric = [0 for _ in range(9)]
        cnt = 0
        for i, x in enumerate(tqdm.tqdm(test_dataloader)):
            for k in x.keys():
                x[k] = x[k].to(local_rank)
                
            d1, d2 = model(x)
            
            gt = x["depth_values"]
            d1 = F.interpolate(d1[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
            d2 = F.interpolate(d2[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
            
            metric = get_metrics(gt, (d1 + d2)/2)
            for i in range(9): tot_metric[i] += metric[i].cpu().detach().item()
            cnt+=1

        for i in range(9): tot_metric[i]/=cnt
        print(tot_metric)
        with open("metric.csv", mode='a', newline='') as file:  # Open in append mode
            writer = csv.writer(file)
            writer.writerow(tot_metric)  # Write the new row only
        
    dist.destroy_process_group()

def run_ddp(world_size):
    # We spawn the processes for each GPU using Python's multiprocessing
    mp.spawn(main, nprocs=world_size, args=(world_size,))

if __name__ == '__main__':
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    world_size = torch.cuda.device_count()
    run_ddp(world_size)