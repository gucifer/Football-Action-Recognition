import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NLLLoss(torch.nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, labels, output):
        # return torch.mean(labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output))
        loss = torch.mean(torch.mean(labels * -torch.log(output + 1e-10) + (1 - labels) * -torch.log(1 - output + 1e-10)))

        return loss


def reweight(cls_num_dict, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_dict: a dict containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None

    per_cls_weights = [(1 - beta)/(1 - np.power(beta, cls_num_dict[i])) for i in range(18)]
    per_cls_weights = torch.asarray(per_cls_weights)
    per_cls_weights = per_cls_weights/torch.sum(per_cls_weights) * 18

    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.9999):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = torch.tensor(weight).float()

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        labels = target

        loss = torch.mean(torch.mean(input * -torch.log(target + 1e-10) + (1 - input) * -torch.log(1 - target + 1e-10)))

        sigmoid = torch.log(1 + torch.exp(-1.0 * input))
        if self.gamma != 0.0:
            loss = torch.exp(-self.gamma * (labels * input + sigmoid))*loss
        weights = self.weight[target.argmax(axis=1)]
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, input.shape[1])
        weighted_loss = weights*loss

        focal_loss = torch.sum(weighted_loss)
        loss = focal_loss/torch.sum(labels)

        return loss
