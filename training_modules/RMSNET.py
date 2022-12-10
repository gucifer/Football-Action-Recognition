import torch
import torch.nn as nn

# Not perferct 

class RMSNetModel(nn.Module):
    def __init__(self, feature_size, num_frames, num_classes, dropout):
        super(RMSNetModel,self).__init__()
        self.feature_size = feature_size
        self.fc = nn.Linear(feature_size*num_frames, 512*num_frames)
        self.temporal_conv1 = torch.nn.Conv1d(in_channels=512, out_channels=512, kernel_size=9, stride=1, padding=4).cuda()
        self.temporal_conv2 = torch.nn.Conv1d(in_channels=512, out_channels=256, kernel_size=9, stride=1, padding=4).cuda()
        self.fc2 = torch.nn.Linear(256, 128).cuda()

        self.fc_class = torch.nn.Linear(128, num_classes+1).cuda()


    def forward(self,x):
        bs, frames_per_event = x.shape[0], x.shape[1]
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(bs, frames_per_event, -1)
        x = torch.nn.functional.relu(self.temporal_conv1(x.permute(0, 2, 1)))
        x = torch.nn.functional.relu(self.temporal_conv2(x))
        x = torch.nn.functional.dropout(x, p=0.1)

        x = x.permute(0, 2, 1)
        x = x.contiguous()

        x_event = torch.max(x, dim=1)[0]
        x_features = self.fc2(x_event)
        out = torch.nn.functional.sigmoid(self.fc_class(x_features))

        return out
