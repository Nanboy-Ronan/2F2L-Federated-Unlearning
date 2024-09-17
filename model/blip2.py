import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess


class BLIP2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model, _, _ = load_model_and_preprocess("blip2_feature_extractor", "pretrain", is_eval=True)
        self.feat_dim = 768

    def forward(self, images):
        sample = {"image": images, "text_input": None}
        return self.model.extract_features(sample, mode="image").image_embeds[:, 0, :]


class BLIP2LP(nn.Module):
    def __init__(self, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder, _, _ = load_model_and_preprocess("blip2_feature_extractor", "pretrain", is_eval=True)
        self.feat_dim = 768
        
        self.num_classes = num_classes

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.head = torch.nn.Linear(self.feat_dim, num_classes)
    
    def jvp(self, model):
        pass
    
    def freeze_jvp(self):
        pass
    
    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        sample = {"image": x, "text_input": None}
        y = self.encoder.extract_features(sample, mode="image").image_embeds[:, 0, :]
        output = self.head(y)
        jvp = self.head(y)
        return output, jvp



