import torch
import torch.nn as nn
import numpy as np

class DN_to_depth(nn.Module):
    """Layer to transform distance and normal into depth
    """
    def __init__(self, batch_size, height, width):
        super(DN_to_depth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False) # 2, H, W  

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False) # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

    def forward(self, norm_normal, distance, inv_K):
        normalized_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        normalized_points = normalized_points.reshape(self.batch_size, 3, self.height, self.width)
        normal_points = (norm_normal * normalized_points).sum(1, keepdim=True)
        depth = distance / (normal_points + 1e-7)
        return depth.abs()