import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassBalanced_CELoss(nn.Module):
    """
    https://arxiv.org/abs/1901.05555
    """
    def __init__(self, reference_labels, num_class, beta=0.999):
        super(ClassBalanced_CELoss, self).__init__()
        self.beta = beta

        self.counts_cls = self.__count_per_class(reference_labels, num_class)
        self.counts_cls = nn.Parameter(torch.from_numpy(np.array(self.counts_cls).astype('float32')), 
                                       requires_grad =False).cuda()
        self.w = self.calc_weight(self.beta) if beta is not None else None

        return

    def __count_per_class(self, labels, num_class):
        unique_labels, count = np.unique(labels, return_counts=True)
        c_per_cls = np.zeros(num_class)
        c_per_cls[unique_labels] = count
        return c_per_cls

    def calc_weight(self, beta):
        """
        Args:
            beta : float or tensor(batch size, 1)
        """
        # effective number
        ef_Ns = (1 - torch.pow(beta, self.counts_cls)) / (1 - beta)

        # weight
        w = 1 / ef_Ns
        # normalize
        if len(w.size()) == 1:
            #WN = torch.mean(w * self.counts_cls)
            W = torch.sum(w)
        else:
            #WN = torch.mean(w * self.counts_cls, dim=1, keepdim=True)
            W = torch.sum(w, dim=1, keepdim=True)
        #N = torch.mean(self.counts_cls)
        C = self.counts_cls.size()[0]
        #w = w * N / WN
        w = w * C / W
        return w
    
    def forward(self, input, label, beta=None):
        """
        Args:
            beta : shape (batch size, 1) or (1, 1) in training, (1, 1) in test
        """
        if beta is None:
            w = self.w[label].unsqueeze(1) # (batch size, 1)
        else:
            w = self.calc_weight(beta) # (batch size, num class) or (1, num class)
            if w.size()[0] == 1 and label.size()[0] != 1:
                w = w.expand(label.size()[0], w.size()[1])
            w = torch.gather(w, -1, label.unsqueeze(1)) # (batch size, 1)

        logp = F.log_softmax(input, dim=-1) # (batch size, num class)
        logp = torch.gather(logp, -1, label.unsqueeze(1)) # (batch size, 1)

        loss = torch.mean(- w * logp)
        return loss