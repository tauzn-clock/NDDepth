import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Swinv2Config, UperNetConfig, UperNetForSemanticSegmentation
from newcrf_layers import NewCRFChain
from BasicUpdateBlockDepth import BasicUpdateBlockDepth
from DN_to_depth import DN_to_depth

class ModelConfig():
    def __init__(self, version):
        super(ModelConfig, self).__init__()
        
        if version[:-2] == 'base':
            self.embed_dim = 128
            self.depths = [2, 2, 18, 2]
            self.num_heads = [4, 8, 16, 32]
            self.in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            self.embed_dim = 192
            self.depths = [2, 2, 18, 2]
            self.num_heads = [6, 12, 24, 48]
            self.in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            self.embed_dim = 96
            self.depths = [2, 2, 6, 2]
            self.num_heads = [3, 6, 12, 24]
            self.in_channels = [96, 192, 384, 768]
        elif version[:-2] == 'micro':
            self.embed_dim = 64
            self.depths = [2, 2, 4, 2]
            self.num_heads = [2, 4, 8, 16]
            self.in_channels = [64, 128, 256, 512]
            
        self.win = 7
        self.crf_dims = [128, 256, 512, 1024]
        self.v_dims = [64, 128, 256, 512]
        
        backbone_config = Swinv2Config(
                embed_dim=self.embed_dim,
                depths=self.depths,
                num_heads=self.num_heads,
                out_features=["stage1", "stage2", "stage3", "stage4"]
                )
        
        self.uper_config = UperNetConfig(backbone_config=backbone_config)


class Model(nn.Module):
    def __init__(self, config):        
        super(Model, self).__init__()
        
        self.config = config
        
        self.backbone = UperNetForSemanticSegmentation(self.config.uper_config)
        
        #self.backbone.eval() # TODO: Necessary to allow evaluation as some layer requies mean??
    
        self.crf_chain_1 = NewCRFChain(self.config.in_channels, self.config.crf_dims, self.config.v_dims, self.config.win)
        self.depth_head = DistanceHead(self.config.crf_dims[0])
        self.uncer_head_1 = UncerHead(self.config.crf_dims[0])
        
        self.crf_chain_2 = NewCRFChain(self.config.in_channels, self.config.crf_dims, self.config.v_dims, self.config.win)
        self.dist_head = DistanceHead(self.config.crf_dims[0])
        self.normal_head = NormalHead(self.config.crf_dims[0])
        self.uncer_head_2 = UncerHead(self.config.crf_dims[0])
        
        self.update = BasicUpdateBlockDepth(context_dim=self.config.in_channels[0])

        #self.dn_to_depth = DN_to_depth(self.config.batch_size, self.config.height, self.config.width)

    def forward(self, x):
        outputs = self.backbone.backbone.forward_with_filtered_kwargs(**x)
        
        features = outputs.feature_maps

        #logits = self.backbone.decode_head(features)
        #logits = nn.functional.interpolate(logits, size=x["pixel_values"].shape[2:], mode="bilinear", align_corners=False)
        
        #laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.backbone .decode_head.lateral_convs)]
            
        psp_out =self.backbone.decode_head.psp_forward(features)
        
        # Depth
        crf_out_1 = self.crf_chain_1(psp_out, features)     
        d1 = self.depth_head(crf_out_1)
        u1 = self.uncer_head_1(crf_out_1)
        
        # Normal Distance
        crf_out_2 = self.crf_chain_2(psp_out, features)
        distance = self.dist_head(crf_out_2)
        n2 = self.normal_head(crf_out_2)

        b, _, h, w =  n2.shape 
        device = n2.device  
        dn_to_depth = DN_to_depth(b, h, w).to(device) # DX: Layer to converts normal + distance to depth

        n2 = F.normalize(n2, dim=1, p=2)
        d2 = dn_to_depth(n2, distance, x["camera_intrinsics_resized"]).clamp(0, 1)
        u2 = self.uncer_head_2(crf_out_2)

        # Iterative refinement
        context = features[0]
        gru_hidden = torch.cat((crf_out_1, crf_out_2), 1)
        depth1_list, depth2_list  = self.update(d1, u1, d2, u2, context, gru_hidden)

        # Resize
        _, _, a, b = x["pixel_values"].shape
        max_depth = x["max_depth"].view(-1, 1, 1, 1)
        for i in range(len(depth1_list)): depth1_list[i] = F.interpolate(depth1_list[i], size=(a,b), mode='bilinear', align_corners=False) * max_depth
        for i in range(len(depth2_list)): depth2_list[i] = F.interpolate(depth2_list[i], size=(a,b), mode='bilinear', align_corners=False) * max_depth
        u1 = F.interpolate(u1, size=(a,b), mode='bilinear', align_corners=False)
        u2 = F.interpolate(u2, size=(a,b), mode='bilinear', align_corners=False)
        n2 = F.interpolate(n2, size=(a,b), mode='bilinear', align_corners=False)
        distance = F.interpolate(distance, size=(a,b), mode='bilinear', align_corners=False)

        return depth1_list, u1, depth2_list, u2, n2, distance

class DistanceHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DistanceHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        return x
    
class NormalHead(nn.Module):
    def __init__(self, input_dim=100):
        super(NormalHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 3, 3, padding=1)
       
    def forward(self, x):
        x = self.conv1(x)
        return x
    

class UncerHead(nn.Module):
    def __init__(self, input_dim=100):
        super(UncerHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        return x