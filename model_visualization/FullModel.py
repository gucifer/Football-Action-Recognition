import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

from Visualization.attention_model import AttentionModel
from torchvision.models.feature_extraction import create_feature_extractor


class FullModel(nn.Module):
    def __init__(self, feature_size, num_frames, num_heads, num_classes, dropout, device):
        self.device = device
        self.attn_model = AttentionModel(feature_size, num_frames, num_heads, num_classes, dropout)
        model_weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        model = models.vit_b_16(weights=model_weights, progress=True).to(device)
        model.eval()
        return_nodes = {"encoder.layers.encoder_layer_11": "encode11"}
        self.feat_model = create_feature_extractor(model, return_nodes)

    def forward(self, loader):
        all_features = []
        for it, trans_attrs in enumerate(loader):
            trans_attrs = trans_attrs.to(self.device)
            features = self.feat_model(trans_attrs)
            encode_feats = features["encode11"][:, 0]
            all_features.append(encode_feats.cpu())
        all_features = np.concatenate(all_features)
        y = self.attn_model(all_features)
        return y
