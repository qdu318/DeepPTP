# encoding utf-8
import torch.nn as nn
import torch
import torch.nn.functional as F


class SpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpatialConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.linear = nn.Linear(2, 16)

    def forward(self, parameters):
        X, Y = torch.unsqueeze(parameters['X'], dim=2), torch.unsqueeze(parameters['Y'], dim=2)
        locations = torch.cat((X, Y), 2)
        locations_linear = F.tanh(self.linear(locations)).permute(0, 2, 1)
        out = F.elu(self.conv(locations_linear))
        return out