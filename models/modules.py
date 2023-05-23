import torch
import torch.nn as nn
import torch.nn.functional as F

class Projector_SimCLR(nn.Module):
    '''
        The projector for SimCLR. This is added on top of a backbone for SimCLR Training
    '''

    def __init__(self, in_dim = 512, out_dim = 512, mid_dim = None, bn = False):
        super(Projector_SimCLR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mid_dim = mid_dim if mid_dim is not None else self.in_dim
        if bn:
            layers = [nn.Linear(self.in_dim, self.mid_dim),
                      nn.BatchNorm1d(self.mid_dim),
                      nn.ReLU(inplace = True),
                      nn.Linear(self.mid_dim, self.out_dim)]
        else:
            layers = [nn.Linear(self.in_dim, self.mid_dim),
                      nn.ReLU(inplace=True),
                      nn.Linear(self.mid_dim, self.out_dim)]
        self.projector = nn.Sequential(*layers)

    def forward(self, x):
        x = self.projector(x)
        return x


class Projector_Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Projector_Linear, self).__init__()
        self.projector = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.projector(x)


class TransConv(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(TransConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_dim),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore


    def forward(self,f_stu, out_shape = None, mode = "nearest"):
        f = self.conv1(f_stu)
        f = F.interpolate(f, size = (out_shape, out_shape), mode = mode)
        f = self.conv2(f)

        return f