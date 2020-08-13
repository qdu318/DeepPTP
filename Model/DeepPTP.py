# encoding utf-8
import torch
import torch.nn as nn
from Model.TCN import TemporalConvNet
from Model.SpatialConv import SpatialConv
import torch.nn.functional as F


class DeepPTP(nn.Module):
    def __init__(self, in_channels_S, out_channels_S, kernel_size_S, num_inputs_T, num_channels_T, num_outputs_T):
        super(DeepPTP, self).__init__()
        self.SpatialConv = SpatialConv(in_channels=in_channels_S, out_channels=out_channels_S, kernel_size=kernel_size_S)
        self.TemporalConv = TemporalConvNet(num_inputs=num_inputs_T + 2, num_channels=num_channels_T)
        self.implicit_net = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size_S)
        self.full_connect = nn.Linear(num_channels_T[-1], num_outputs_T)

    def forward(self, parameters):
        out = self.SpatialConv(parameters)

        distance = torch.unsqueeze(parameters['distance'], dim=2).permute(0, 2, 1)
        distance = self.implicit_net(distance).permute(0, 2, 1)
        out = torch.cat((out.permute(0, 2, 1), distance), dim=2)

        times = torch.unsqueeze(parameters['time_gap'], dim=2).permute(0, 2, 1)
        times = self.implicit_net(times).permute(0, 2, 1)
        out = torch.cat((out, times), dim=2).permute(0, 2, 1)

        out = self.TemporalConv(out)
        out = self.full_connect(out[:, :, -1])

        return F.log_softmax(out, dim=1)
