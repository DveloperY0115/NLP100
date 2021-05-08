import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import SimpleNet
from data_loader import process_data
from data_loader import create_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--n_batch', type=int, default=32)
parser.add_argument('--num_worker', type=int, default=0)
parser.add_argument('--num_epoch', type=int, default=100)

args = parser.parse_args()


def run_train(train_dataloader, net, optimizer, criterion):

    total_loss = 0
    n_data = len(train_dataloader)

    pbar = tqdm(total=n_data, leave=False)

    for i, sample in enumerate(train_dataloader):

        optimizer.zero_grad()

        X, y = sample[0], sample[1]

        X = X.type(torch.float)
        y = torch.argmax(y, dim=1)

        loss = criterion(net(X), y)
        
        # back propagation
        loss.backward()
        optimizer.step()

        total_loss += loss

        pbar.update(1)

    pbar.close()

    # return mean loss
    return total_loss / n_data


def run_valid(valid_dataloader, net):
    pass


def run_test(test_dataloader, net):
    pass


def train_one_epoch(train_dataloader, valid_dataloader, model, optimizer, criterion):
    mean_loss = run_train(train_dataloader, model, optimizer, criterion)

    with torch.no_grad():
        run_valid(valid_dataloader, model)

    return mean_loss


def main():
    # process data
    train_X, train_y, valid_X, valid_y, test_X, test_y = process_data()

    # prepare datasets & loaders
    train_dataset = create_dataset(train_X, train_y)
    valid_dataset = create_dataset(valid_X, valid_y)
    test_dataset = create_dataset(test_X, test_y)

    train_dataloader = DataLoader(train_dataset, batch_size=args.n_batch, shuffle=True, num_workers=args.num_worker)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.n_batch, shuffle=True, num_workers=args.num_worker)
    test_dataloader = DataLoader(test_dataset, batch_size=args.n_batch, shuffle=True, num_workers=args.num_worker)

    # initialize model: input sequence length is 29 (just set)
    net = SimpleNet(29)

    # initialize optimizer
    optimizer = optim.Adam(net.parameters())

    # criterion
    criterion = nn.CrossEntropyLoss()

    for i in range(args.num_epoch):
        train_one_epoch(train_dataloader, valid_dataloader, net, optimizer, criterion)

    run_test(test_dataloader, net)


if __name__ == '__main__':
    main()
