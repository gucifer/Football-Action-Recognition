import torch
import torch.nn as nn
from .netvlad import NetVLAD


class AttentionModel(nn.Module):
    def __init__(self, feature_size, num_frames, num_heads, num_classes, dropout):
        super(AttentionModel,self).__init__()
        self.feature_size = feature_size

        self.attention_1 = nn.MultiheadAttention(self.feature_size,num_heads,dropout=dropout, batch_first=True)
        self.linear_net = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.feature_size),
        )
        self.final_layer = nn.Linear(self.feature_size, 128)
        self.pos_embedding = nn.Embedding(num_frames, self.feature_size).cuda()

        self.sigm = nn.Sigmoid()
        self.norm_1 = nn.LayerNorm(normalized_shape=(num_frames, feature_size))
        self.f_class = nn.Linear(num_frames*128, num_classes+1)
        # self.global_pooling = nn.GlobalAveragePooling()
        
        # NEtvlad model
        self.vlad_k = 64 # Size of the vocabulary for NetVLAD
        self.drop_1 = nn.Dropout(0.2)
        self.drop_2 = nn.Dropout(0.3)
        self.drop_3 = nn.Dropout(0.4)
        self.pool_layer_before = NetVLAD(cluster_size=int(self.vlad_k/2), feature_size=feature_size,
                                            add_batch_norm=True)
        self.pool_layer_after = NetVLAD(cluster_size=int(self.vlad_k/2), feature_size=feature_size,
                                        add_batch_norm=True)
        self.fc = nn.Linear(feature_size*self.vlad_k, num_classes+1)


    def forward(self,x):
        out = self.embed(x)
        out = self.drop_1(out)
        out, attn = self.attention_1(out,out,out)
        out = self.drop_2(out)
        # out = self.linear_net(out)
        # res_out = x + out
        # out = x * out
        # out = self.norm_1(out)

        # For normal attention model
        # out = self.global_pooling(out)
        # out = self.final_layer(out)
        # # out = nn.Flatten()(out)
        # # out = self.linear_net(out)
        # out = self.sigm(self.f_class(out))

        # Netvlad hybrid
        # inputs_pooled = self.pool_layer(out)
        # out = self.sigm(self.fc(self.drop_2(inputs_pooled)))

        nb_frames_50 = int(out.shape[1]/2)
        inputs_before_pooled = self.pool_layer_before(out[:, :nb_frames_50, :])
        inputs_after_pooled = self.pool_layer_after(out[:, nb_frames_50:, :])
        out = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)
        out = self.sigm(self.fc(self.drop_3(out)))

        return out

    def embed(self, inputs):
        embeddings = None
        positions = torch.tensor([[i for i in range(inputs.shape[1])]]*inputs.shape[0]).cuda()
        embeddings = inputs + self.pos_embedding(positions)
        return embeddings

