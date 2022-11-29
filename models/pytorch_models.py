import torch
from torch import nn


class Permute(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (0, 2, 1))


class MaxOverTime(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.max(x, 2).values


class AlphaModel(nn.Module):

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        

        self.alpha_model = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            Permute(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=9, stride=1),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=9, stride=1),
            nn.Dropout(p=0.1),
            MaxOverTime(),
            nn.Linear(in_features=128, out_features=64)
        )

        self.classification_model = nn.Sequential(
            nn.Linear(in_features=64, out_features=self.num_classes),
            # nn.Softmax(dim=1)  # CE loss ahead already has log softmax
        )

        self.regression_model = nn.Sequential(
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        fc_out = self.alpha_model(x)
        out = self.classification_model(fc_out)
        pred_rel_offset = self.regression_model(fc_out)
        return out, pred_rel_offset