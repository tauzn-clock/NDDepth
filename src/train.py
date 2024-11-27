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
from DN_to_distance import DN_to_distance
from loss import silog_loss, get_metrics
from segmentation import compute_seg, get_smooth_ND

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
    
    BATCH_SIZE = 8

    train_dataset = BaseImageDataset('train', NYUImageData, '/scratchdata/nyu_depth_v2/sync', '/NDDepth/src/nyu_train.csv')
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, sampler=train_sampler)

    test_dataset = BaseImageDataset('test', NYUImageData, '/scratchdata/nyu_depth_v2/official_splits/test', '/NDDepth/src/nyu_test.csv')
    test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True, sampler=test_sampler)

    csv_file = [["silog", "abs_rel", "log10", "rms", "sq_rel", "log_rms", "d1", "d2", "d3"]]
    with open('metric.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_file)

    config =  ModelConfig("tiny07")
    config.batch_size = BATCH_SIZE
    config.height = 480//4
    config.width = 640//4
    model = Model(config).to(local_rank)
    model.backbone.backbone.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        model.train()
        silog_criterion = silog_loss(variance_focus=0.85)
        dn_to_distance = DN_to_distance(config.batch_size, config.height * 4, config.width * 4).to(local_rank)
        loop = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch")
        for _, x in enumerate(loop):
            optimizer.zero_grad()
            for k in x.keys():
                x[k] = x[k].to(local_rank)

            d1_list, u1, d2_list, u2, norm_est, dist_est = model(x)

            # Post Processing

            gt = x["depth_values"]
            normal_gt = torch.stack([x["normal_values"][:, 0], x["normal_values"][:, 2], x["normal_values"][:, 1]], 1).to(local_rank)
            normal_gt_norm = F.normalize(normal_gt, dim=1, p=2).to(local_rank)
            distance_gt = dn_to_distance(gt, normal_gt_norm, x["camera_intrinsics"])

            # Depth Loss

            loss_depth1_0 = silog_criterion(d1_list[0], gt, x["mask"])
            loss_depth2_0 = silog_criterion(d2_list[0], gt, x["mask"])

            loss_depth1 = 0
            loss_depth2 = 0
            weights_sum = 0
            for i in range(len(d1_list) - 1):
                loss_depth1 += (0.85**(len(d1_list)-i-2)) * silog_criterion(d1_list[i + 1], gt, x["mask"])
                loss_depth2 += (0.85**(len(d2_list)-i-2)) * silog_criterion(d2_list[i + 1], gt, x["mask"])
                weights_sum += 0.85**(len(d1_list)-i-2)
            
            loss_depth = 10 * ((loss_depth1 + loss_depth2) / weights_sum + loss_depth1_0 + loss_depth2_0 )
            
            # Uncertainty Loss

            uncer1_gt = torch.exp(-5 * torch.abs(gt - d1_list[0].detach()) / (gt + d1_list[0].detach() + 1e-7))
            uncer2_gt = torch.exp(-5 * torch.abs(gt - d2_list[0].detach()) / (gt + d2_list[0].detach() + 1e-7))
            
            loss_uncer1 = torch.abs(u1-uncer1_gt)[x["mask"]].mean()
            loss_uncer2 = torch.abs(u2-uncer2_gt)[x["mask"]].mean()

            loss_uncer = loss_uncer1 + loss_uncer2

            loss_normal = 5 * ((1 - (normal_gt_norm * norm_est).sum(1, keepdim=True))[x["mask"]]).mean() #* x["mask"]).sum() / (x["mask"] + 1e-7).sum()
            loss_distance = 0.25 * torch.abs(distance_gt- dist_est)[x["mask"]].mean()

            # Segmentation Loss
            segment, planar_mask, dissimilarity_map = compute_seg(x["pixel_values"], norm_est, dist_est[:, 0])
            loss_grad_normal, loss_grad_distance = get_smooth_ND(norm_est, dist_est, planar_mask)

            loss_seg = 0.01 * (loss_grad_distance + loss_grad_normal)

            loss = loss_depth + loss_uncer + loss_normal + loss_distance + loss_seg
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            
            #loop.set_postfix(loss=loss.item())
            custom_message = "Depth: {:.3g}, ".format(loss_depth.item())
            custom_message += "Uncer: {:.3g}, ".format(loss_uncer.item())
            custom_message += "Normal: {:.3g}, ".format(loss_normal.item())
            custom_message += "Dist: {:.3g}, ".format(loss_distance.item())
            custom_message += "Seg: {:.3g}".format(loss_seg.item())
            loop.set_postfix(message=custom_message)
        # Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9999
        print(param_group['lr'])
        torch.save(model.module.state_dict(), 'model.pth')
        
        model.eval()
        torch.cuda.empty_cache()
        tot_metric = [0 for _ in range(9)]
        cnt = 0
        with torch.no_grad():
            for _, x in enumerate(tqdm.tqdm(test_dataloader)):
                #if x["pixel_values"].shape[0]!=BATCH_SIZE: break
                for k in x.keys():
                    x[k] = x[k].to(local_rank)
                    
                d1_list, _, d2_list, _, _, _ = model(x)
                
                gt = x["depth_values"]
                d1 = d1_list[-1]
                d2 = d2_list[-1]
                
                metric = get_metrics(gt, (d1 + d2)/2, x["mask"])
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