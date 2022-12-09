import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, feature_size, num_frames, num_heads, num_classes, dropout):
        super(AttentionModel,self).__init__()
        self.feature_size = feature_size
        self.attention_1 = nn.MultiheadAttention(self.feature_size,num_heads,dropout=dropout)
        self.linear_net = nn.Sequential(
            nn.Linear(feature_size*num_frames, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            # nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes+1)
        )

        self.sigm = nn.Sigmoid()
        self.norm_1 = nn.LayerNorm(normalized_shape=(num_frames, feature_size))
        self.norm_2 = nn.LayerNorm(normalized_shape=(num_frames, feature_size))


    def forward(self,x):
        out = self.norm_1(x)
        out, attn = self.attention_1(out,out,out)
        res_out = x + out
        out = self.norm_2(res_out)
        out = nn.Flatten()(out)
        # out = self.linear_net(out)
        out = self.sigm(self.linear_net(out))
        return out
