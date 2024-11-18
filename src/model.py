import torch
from torch import nn

from transformers import Swinv2Config, UperNetConfig, UperNetForSemanticSegmentation
from newcrf_layers import NewCRFChain

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
        
        self.backbone.eval() # TODO: Necessary to allow evaluation as some layer requies mean??
    
        self.crf_chain_1 = NewCRFChain(self.config.in_channels, self.config.crf_dims, self.config.v_dims, self.config.win)
        self.dist_head_1 = DistanceHead(self.config.crf_dims[0])
        self.uncer_head_1 = UncerHead(self.config.crf_dims[0])
        
    def forward(self, x):
        outputs = self.backbone.backbone.forward_with_filtered_kwargs(**x)
        
        features = outputs.feature_maps

        logits = self.backbone.decode_head(features)
        logits = nn.functional.interpolate(logits, size=x.pixel_values.shape[2:], mode="bilinear", align_corners=False)
        
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.backbone .decode_head.lateral_convs)]
            
        psp_out =self.backbone.decode_head.psp_forward(features)
        
        crf_out_1 = self.crf_chain_1(psp_out, features)     
        d1 = self.dist_head_1(crf_out_1)
        u1 = self.uncer_head_1(crf_out_1)

        return None

class DistanceHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DistanceHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        return x

class UncerHead(nn.Module):
    def __init__(self, input_dim=100):
        super(UncerHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        return x