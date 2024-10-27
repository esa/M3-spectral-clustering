import torch
import torch.nn as nn
from torch.autograd import Variable


class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        # Constructor
        super(GaussianDropout, self).__init__()

        self.alpha = torch.Tensor([alpha])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x


class ASC(nn.Module):
    def __init__(self):
        super(ASC, self).__init__()

    def forward(self, input):
        """Abundances Sum-to-One Constraint"""
        constrained = input/torch.sum(input, dim=0)*100
        return constrained


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x
