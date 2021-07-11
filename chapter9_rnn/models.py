import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):
    def __init__(
        self, num_layers, hidden_dim, num_vocab, embed_dim, num_classes, dropout_p=0.5, **kwargs
    ):
        """
        Constructor of SimpleRNN

        Args:
        - embed_args (tuple). A tuple containing arguments for nn.Embedding
        - rnn_args (tuple). A tuple containing arguments for nn.RNN
        """

        super(SimpleRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # embedding layer
        self.embed = nn.Embedding(num_vocab, embed_dim)

        # basic RNN
        self.recurrent = nn.RNN(embed_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)

        # fully connected layers for texts
        self.fc_1 = nn.Linear(hidden_dim, 32)
        self.fc_2 = nn.Linear(32, 16)
        self.fc_3 = nn.Linear(16, num_classes)

        self.fc_4 = nn.Linear(kwargs["num_publisher"], 32)
        self.fc_5 = nn.Linear(32, 16)
        self.fc_6 = nn.Linear(16, num_classes)

        # drop-out layer
        self.do = nn.Dropout(dropout_p)

    def forward(self, x, z):
        """
        Forward propagation.

        Args:
        - x: Batch of texts.
        - z: Batch of publisher labels.
        """

        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.recurrent(x, h_0)
        x = x[:, -1, :]
        x = self.do(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))

        z = F.relu(self.fc_4(z.type(torch.float)))
        z = F.relu(self.fc_5(z))
        z = F.relu(self.fc_6(z))

        out = x + z
        return out

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()


class SimpleLSTM(nn.Module):
    def __init__(
        self, num_layers, hidden_dim, num_vocab, embed_dim, num_classes, dropout_p=0.5, **kwargs
    ):

        super(SimpleLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # embedding layer
        self.embed = nn.Embedding(num_vocab, embed_dim)

        # LSTM
        self.recurrent = nn.LSTM(
            embed_dim, hidden_dim, num_layers=self.num_layers, batch_first=True
        )

        # fully connected layers for texts
        self.fc_1 = nn.Linear(hidden_dim, 32)
        self.fc_2 = nn.Linear(32, 16)
        self.fc_3 = nn.Linear(16, num_classes)

        self.fc_4 = nn.Linear(kwargs["num_publisher"], 32)
        self.fc_5 = nn.Linear(32, 16)
        self.fc_6 = nn.Linear(16, num_classes)

        # drop-out layer
        self.do = nn.Dropout(dropout_p)

    def forward(self, x, z):
        """
        Forward propagation.

        Args:
        - x: Batch of texts.
        - z: Batch of publisher labels.
        """

        x = self.embed(x)
        h_0, c_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.recurrent(x, (h_0, c_0))
        x = x[:, -1, :]
        x = self.do(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))

        z = F.relu(self.fc_4(z.type(torch.float)))
        z = F.relu(self.fc_5(z))
        z = F.relu(self.fc_6(z))

        out = x + z
        return out

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return (
            weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
        )


class SimpleGRU(nn.Module):
    def __init__(
        self, num_layers, hidden_dim, num_vocab, embed_dim, num_classes, dropout_p=0.5, **kwargs
    ):

        super(SimpleGRU, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # embedding layer
        self.embed = nn.Embedding(num_vocab, embed_dim)

        # LSTM
        self.recurrent = nn.GRU(embed_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)

        # fully connected layers for texts
        self.fc_1 = nn.Linear(hidden_dim, 32)
        self.fc_2 = nn.Linear(32, 16)
        self.fc_3 = nn.Linear(16, num_classes)

        self.fc_4 = nn.Linear(kwargs["num_publisher"], 32)
        self.fc_5 = nn.Linear(32, 16)
        self.fc_6 = nn.Linear(16, num_classes)

        # drop-out layer
        self.do = nn.Dropout(dropout_p)

    def forward(self, x, z):
        """
        Forward propagation.

        Args:
        - x: Batch of texts.
        - z: Batch of publisher labels.
        """

        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.recurrent(x, h_0)
        x = x[:, -1, :]
        x = self.do(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))

        z = F.relu(self.fc_4(z.type(torch.float)))
        z = F.relu(self.fc_5(z))
        z = F.relu(self.fc_6(z))

        out = x + z
        return out

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()

