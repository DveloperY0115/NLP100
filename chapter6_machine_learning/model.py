import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):

    def __init__(self, seq_len, n_batch,
                 input_size=1, hidden_size=15, num_layers=2, num_classes=4):
        super(SimpleNet, self).__init__()

        self.seq_len = seq_len
        self.n_batch = n_batch
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=False)

        self.fc1 = nn.Linear(self.hidden_size, 50)
        self.fc2 = nn.Linear(50, self.num_classes)

    def forward(self, x):

        x = x.transpose(0, 1)
        x = x.unsqueeze(-1)

        # initialize hidden state & cell states
        h0 = torch.randn(self.num_layers, self.n_batch, self.hidden_size)
        c0 = torch.randn(self.num_layers, self.n_batch, self.hidden_size)

        x, (_, _) = self.lstm(x, (h0, c0))
        x = x[-1]
        x = self.fc1(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
