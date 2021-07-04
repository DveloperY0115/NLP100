import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):

    def __init__(self, num_layers=2, num_category=4):

        super().__init__(SimpleRNN, self)
        
        # basic RNN
        self.recurrent = nn.RNN(input_size=128, hidden_size=32,
                            num_layers=num_layers, batch_first=True)
        
        # fully connected layers for texts
        self.fc_1 = nn.Linear(32, 16)
        self.fc_2 = nn.Linear(16, num_category)

        self.fc_3 = nn.Linear(5, 32)
        self.fc_4 = nn.Linear(32, 16)
        self.fc_5 = nn.Linear(16, num_category)


    def forward(x, z):
        """
        Forward propagation.

        Args:
        - x: Batch of texts.
        - z: Batch of publisher labels.
        """
        x = self.recurrent(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))

        z = F.relu(self.fc_3(z))
        z = F.relu(self.fc_4(z))
        z = F.relu(self.fc_5(z))

        out = F.softmax(x + z)
        return out


class SimpleLSTM(nn.Module):

    def __init__(self, num_layers=2, num_category=4):

        super().__init__(SimpleLSTM, self)

        # LSTM
        self.recurrent = nn.LSTM(input_size=128, hidden_size=32,
                            num_layers=num_layers, batch_first=True)
        
        # fully connected layers for texts
        self.fc_1 = nn.Linear(32, 16)
        self.fc_2 = nn.Linear(16, num_category)

        self.fc_3 = nn.Linear(5, 32)
        self.fc_4 = nn.Linear(32, 16)
        self.fc_5 = nn.Linear(16, num_category)


    def forward(x, z):
        """
        Forward propagation.

        Args:
        - x: Batch of texts.
        - z: Batch of publisher labels.
        """
        x = self.recurrent(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))

        z = F.relu(self.fc_3(z))
        z = F.relu(self.fc_4(z))
        z = F.relu(self.fc_5(z))

        out = F.softmax(x + z)
        return out


class SimpleGRU(nn.Module):

    def __init__(self, num_layers=2, num_category=4):

        super().__init__(SimpleGRU, self)

        # LSTM
        self.recurrent = nn.GRU(input_size=128, hidden_size=32,
                            num_layers=num_layers, batch_first=True)
        
        # fully connected layers for texts
        self.fc_1 = nn.Linear(32, 16)
        self.fc_2 = nn.Linear(16, num_category)

        self.fc_3 = nn.Linear(5, 32)
        self.fc_4 = nn.Linear(32, 16)
        self.fc_5 = nn.Linear(16, num_category)

    def forward(x, z):
        """
        Forward propagation.

        Args:
        - x: Batch of texts.
        - z: Batch of publisher labels.
        """
        x = self.recurrent(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))

        z = F.relu(self.fc_3(z))
        z = F.relu(self.fc_4(z))
        z = F.relu(self.fc_5(z))

        out = F.softmax(x + z)
        return out
        