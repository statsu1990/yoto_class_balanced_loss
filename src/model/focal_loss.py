import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target, gamma=None, alpha=None):
        logpt = F.log_softmax(input, dim=-1)
        logpt = torch.gather(logpt, -1, target.unsqueeze(1))

        with torch.no_grad():
            pt = torch.exp(logpt)

        if gamma is None:
            loss = -1 * (1 - pt) ** self.gamma * logpt
        else:
            loss = -1 * (1 - pt) ** gamma * logpt

        if alpha is None:
            loss = torch.mean(self.alpha * loss)
        else:
            loss = torch.mean(alpha * loss)
        return loss