import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from utils.utils import AvgrageMeter, accuracy

class torchModel(nn.Module):
    def __init__(self, config, input_shape=(1, 28, 28), num_classes=10):
        super(torchModel, self).__init__()
        layers = []
        n_layers = config['n_layers']
        n_conv_layers = 1
        kernel_size = 2
        in_channels = input_shape[0]
        out_channels = 4
        # for fair comparison
        self.dropout_prob = 0.2

        for i in range(n_conv_layers):
            c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=2, padding=1
                         )
            a = nn.ReLU(inplace=False)
            p = nn.MaxPool2d(kernel_size=2, stride=1)
            layers.extend([c, a, p])
            in_channels = out_channels
            out_channels *= 2

        self.conv_layers = nn.Sequential(*layers)
        self.output_size = num_classes

        self.fc_layers = nn.ModuleList()
        n_in = self._get_conv_output(input_shape)
        n_out = 256
        for i in range(n_layers):
            fc = nn.Linear(int(n_in), int(n_out))
            self.fc_layers += [fc]
            n_in = n_out
            n_out /= 2

        self.last_fc = nn.Linear(int(n_in), self.output_size)
        self.dropout = nn.Dropout(p=self.dropout_prob)

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = torch.tensor(torch.rand(bs, *shape))
        output_feat = self.conv_layers(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        for fc_layer in self.fc_layers:
            x = self.dropout(F.relu(fc_layer(x)))
        x = self.last_fc(x)
        return x

    