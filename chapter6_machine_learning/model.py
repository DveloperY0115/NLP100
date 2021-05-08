import torch
import torch.nn as nn


class SimpleNet(nn.Module):

    def __init__(self, seq_len, num_classes=4):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(seq_len, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, num_classes)

    def forward(self, X):
        x = self.fc1(X)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
