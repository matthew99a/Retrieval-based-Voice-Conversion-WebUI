import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

bn_momentum = 1e-3

periods = [1, 2, 3, 5, 7, 11]
segment_size = [1939, 1934, 1929, 1865, 1855, 2024]
depth = [5, 4, 4, 3, 3, 3]
channels = [64, 128, 256, 256, 512]
kernel_size_conv = [5, 5, 5, 5, 5]
kernel_size_pool = [3, 3, 3, 3, 3]
fc_width = [3, 7, 3, 9, 5, 2]
class PitchContourDiscriminatorP(nn.Module):
    def __init__(self, c, p, t):
        super(PitchContourDiscriminatorP, self).__init__()
        self.p = p
        self.t = t
        self.blocks = nn.ModuleList([])
        self.pools = nn.ModuleList([])
        for i in range(depth[self.p]):
            self.blocks.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels=c if i == 0 else channels[i - 1], out_channels=channels[i], kernel_size=kernel_size_conv[i], bias=False),
                        nn.BatchNorm1d(channels[i], momentum=bn_momentum),
                        nn.LeakyReLU(),
                        )
                    )
            self.blocks.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels=channels[i], out_channels=channels[i], kernel_size=kernel_size_conv[i], bias=False),
                        nn.BatchNorm1d(channels[i], momentum=bn_momentum),
                        nn.LeakyReLU(),
                        )
                    )
            self.pools.append(nn.MaxPool1d(kernel_size=kernel_size_pool[i]))
        self.fc1 = nn.Sequential(
                nn.Linear(channels[depth[self.p] - 1] * fc_width[self.p], channels[depth[self.p] - 1] // 2),
                nn.BatchNorm1d(channels[depth[self.p] - 1] // 2, momentum=bn_momentum),
                nn.LeakyReLU(),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(channels[depth[self.p] - 1] // 2, channels[depth[self.p] - 1] // 2),
                nn.LeakyReLU(),
                )
        self.fc3 = nn.Sequential(
                nn.Linear(channels[depth[self.p] - 1] // 2, 1),
                nn.Sigmoid(),
                )


    def forward(self, x):
        x = x[:, :, (x.shape[2] + segment_size[self.p]) // 2 - segment_size[self.p]:(x.shape[2] + segment_size[self.p]) // 2]
        x = x.view(x.shape[0], x.shape[1], -1, periods[self.p])
        if self.t:
            x = torch.transpose(x, 2, 3)
        x = torch.transpose(x, 1, 2)
        x = x.reshape(x.shape[0] * periods[self.p], x.shape[2], -1)
        for i in range(depth[self.p]):
            x = self.blocks[i * 2](x)
            x = self.blocks[i * 2 + 1](x)
            x = self.pools[i](x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(x.shape[0] // periods[self.p], periods[self.p])
        return x


class PitchContourDiscriminator(nn.Module):
    def __init__(self, c):
        super(PitchContourDiscriminator, self).__init__()
        self.discs = nn.ModuleList([])
        for i in range(len(periods)):
            self.discs.append(PitchContourDiscriminatorP(c, i, False))
            if periods[i] > 1:
                self.discs.append(PitchContourDiscriminatorP(c, i, True))


    def forward(self, x):
        return torch.cat(tuple(disc(x) for disc in self.discs), dim=1)
