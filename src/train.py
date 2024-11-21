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
from dataloader.BaseDataloader import BaseImageDataset
from dataloader.NYUDataloader import NYUImageData
from loss import silog_loss, get_metrics

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

    train_dataset = BaseImageDataset('train', NYUImageData, '/scratchdata/nyu_data', '/scratchdata/nyu_data/data/nyu2_train.csv')
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    train_dataloader = DataLoader(train_dataset, batch_size=6, pin_memory=True, sampler=train_sampler)

    test_dataset = BaseImageDataset('test', NYUImageData, '/scratchdata/nyu_data', '/scratchdata/nyu_data/data/nyu2_test.csv')
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank)
    test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True, sampler=test_sampler)

    csv_file = [["silog", "abs_rel", "log10", "rms", "sq_rel", "log_rms", "d1", "d2", "d3"]]
    with open('metric.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_file)

    config =  ModelConfig("tiny07")
    model = Model(config).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for _, x in enumerate(tqdm.tqdm(train_dataloader)):
            for k in x.keys():
                x[k] = x[k].to(local_rank)

            d1_list, u1, d2_list, u2 = model(x)

            gt = x["depth_values"]
            for i in range(len(d1_list)): d1_list[i] = F.interpolate(d1_list[i], size=gt.shape[2:], mode='bilinear', align_corners=False)
            for i in range(len(d2_list)): d2_list[i] = F.interpolate(d2_list[i], size=gt.shape[2:], mode='bilinear', align_corners=False)
            u1 = F.interpolate(u1, size=gt.shape[2:], mode='bilinear', align_corners=False)
            u2 = F.interpolate(u2, size=gt.shape[2:], mode='bilinear', align_corners=False)

            uncer1_gt = torch.exp(-5 * torch.abs(gt - d1_list[0].detach()) / (gt + d1_list[0].detach() + 1e-7))
            uncer2_gt = torch.exp(-5 * torch.abs(gt - d2_list[0].detach()) / (gt + d2_list[0].detach() + 1e-7))
            
            loss_uncer1 = torch.abs(u1-uncer1_gt)[x["mask"]].mean()
            loss_uncer2 = torch.abs(u2-uncer2_gt)[x["mask"]].mean()

            loss_depth1_0 = silog_loss(d1_list[0], gt, x["mask"])
            loss_depth2_0 = silog_loss(d2_list[0], gt, x["mask"])

            loss_depth1 = 0
            loss_depth2 = 0
            weights_sum = 0
            for i in range(len(d1_list) - 1):
                loss_depth1 += (0.85**(len(d1_list)-i-2)) * silog_loss(d1_list[i + 1], gt, x["mask"])
                loss_depth2 += (0.85**(len(d2_list)-i-2)) * silog_loss(d2_list[i + 1], gt, x["mask"])
                weights_sum += 0.85**(len(d1_list)-i-2)

            loss = (loss_depth1 + loss_depth2) / weights_sum + loss_depth1_0 + loss_depth2_0 + loss_uncer1 + loss_uncer2
            loss = loss.mean()

            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        # Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9999
        print(param_group['lr'])
        print(f"Epoch {epoch} Loss: {running_loss / len(train_dataloader)}")
        torch.save(model.module.state_dict(), 'model.pth')
        
        model.eval()
        tot_metric = [0 for _ in range(9)]
        cnt = 0
        for _, x in enumerate(tqdm.tqdm(test_dataloader)):
            for k in x.keys():
                x[k] = x[k].to(local_rank)
                
            d1_list, _, d2_list, _ = model(x)
            
            gt = x["depth_values"]
            d1 = F.interpolate(d1_list[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
            d2 = F.interpolate(d2_list[-1], size=gt.shape[2:], mode='bilinear', align_corners=False)
            
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