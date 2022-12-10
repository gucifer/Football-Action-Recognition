import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

from torchvision.models.feature_extraction import create_feature_extractor

class Feats2Clip(nn.Module):

    def __init__(self, stride=15, clip_length=15, padding = "replicate_last", off=0) -> None:
        super().__init__()
        self.stride = stride
        self.clip_length = clip_length
        self.padding = padding
        self.off = off


    def forward(self, feats):
        N, F = feats.shape
        feats = feats.view(-1, self.stride+self.clip_length, F)
        return feats
        
class FullModel(nn.Module):
    def __init__(self, model, device):
        super(FullModel, self).__init__()
        self.classification_model = model.to(device)
        self.model_weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        vit_feat_model = models.vit_b_16(weights=self.model_weights, progress=True).to(device)
        return_nodes = {"encoder.layers.encoder_layer_11": "encode11"}
        self.feature_model = create_feature_extractor(vit_feat_model, return_nodes)
        self.feats2clip_model = Feats2Clip()

    def forward(self, x):
        # out = self.model(x)
        feats = self.feature_model(x)["encode11"][:, 0]
        stacked = self.feats2clip_model(feats)
        out = self.classification_model(stacked)
        return out

    def preprocess(self, x):
        func = self.model_weights.transforms()
        out = func(x)
        return out