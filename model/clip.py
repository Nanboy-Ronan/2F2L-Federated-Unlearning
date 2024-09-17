import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from utils.model_util import *


class CLIP(nn.Module):
    def __init__(self, backbone="ViT-B/32", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model, _ = clip.load(backbone, device="cpu")
        self.feat_dim = 512

    def forward(self, images):
        return self.model.visual(images)

class CLIPLP(nn.Module):
    def __init__(self, num_classes, backbone="ViT-B/32", *args, **kwargs) -> None:
        '''
        We use ViT-B/32 and RN50 for this study.
        '''
        super().__init__(*args, **kwargs)

        self.encoder, _ = clip.load(backbone, device="cpu")
        if backbone == "ViT-B/32":
            self.encoder.feat_dim = 512
        elif backbone == "RN50":
            self.encoder.feat_dim = 1024
        else:
            raise NotImplementedError
        
        self.num_classes = num_classes

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.head = torch.nn.Linear(self.encoder.feat_dim, num_classes)
    
    def jvp(self, model):
        pass
    
    def freeze_jvp(self):
        pass

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        y = self.encoder.visual(x)
        output = self.head(y)
        jvp = self.head(y)
        return output, jvp