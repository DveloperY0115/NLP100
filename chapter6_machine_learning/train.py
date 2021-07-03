import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import SimpleNet
from data_loader import process_data
from data_loader import create_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--n_batch', type=int, default=32)
parser.add_argument('--num_worker', type=int, default=0)
parser.add_argument('--num_epoch', type=int, default=300)
parser.add_argument('--log_dir', type=str, default='outputs')
# parser.add_argument('--checkpoint_dir', type=str, default='./outputs/complete.tar')
parser.add_argument('--checkpoint_dir', type=str, default=None)

args = parser.parse_args()


def run_train(train_dataloader, net, optimizer, criterion):

    total_loss = 0
    n_data = len(train_dataloader)

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

    # return mean loss
    return total_loss / n_data


def run_valid(valid_dataloader, net, criterion):

    total_loss = 0
    correct = 0
    n_data = 0

    for i, sample in enumerate(valid_dataloader):

        X, y = sample[0], sample[1]

        X = X.type(torch.float)
        y = torch.argmax(y, dim=1)

        pred = net(X)
        # calculate loss
        loss = criterion(pred, y)
        total_loss += loss

        # calculate accuracy
        for j in range(pred.size()[0]):
            if torch.argmax(pred[j]) == torch.argmax(y[j]):
                correct += 1

        n_data += pred.size()[0]

    acc = (correct / n_data) * 100

    # return mean loss and accuracy
    return acc, total_loss / n_data


def run_test(test_dataloader, net, criterion):

    total_loss = 0
    correct = 0
    n_data = 0

    for i, sample in enumerate(test_dataloader):
        x, y = sample[0], sample[1]

        x = x.type(torch.float)
        y = torch.argmax(y, dim=1)

        pred = net(x)

        # calculate loss
        loss = criterion(pred, y)
        total_loss += loss

        # calculate accuracy
        for j in range(pred.size()[0]):
            if torch.argmax(pred[j]) == torch.argmax(y[j]):
                correct += 1

        n_data += pred.size()[0]

    acc = (correct / n_data) * 100

    # return mean loss and accuracy
    return acc, total_loss / n_data


def main():
    # process data
    train_X, train_y, valid_X, valid_y, test_X, test_y = process_data()

    # prepare datasets & loaders
    train_dataset = create_dataset(train_X, train_y)
    valid_dataset = create_dataset(valid_X, valid_y)
    test_dataset = create_dataset(test_X, test_y)

    train_dataloader = DataLoader(train_dataset, batch_size=args.n_batch,
                                  shuffle=True, num_workers=args.num_worker, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.n_batch,
                                  shuffle=True, num_workers=args.num_worker, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.n_batch,
                                 shuffle=True, num_workers=args.num_worker, drop_last=True)

    # initialize model: input sequence length is 29 (just set)
    net = SimpleNet(29, args.n_batch)

    # initialize optimizer
    optimizer = optim.Adam(net.parameters())

    # criterion
    criterion = nn.CrossEntropyLoss()

    if args.checkpoint_dir is not None:
        # load model, run test
        checkpoint = torch.load(args.checkpoint_dir)
        net.load_state_dict(checkpoint['model_state_dict'])

        acc, test_loss = run_test(test_dataloader, net, criterion)

    else:
        # configure writer
        writer = SummaryWriter(log_dir=args.log_dir)

        for i in tqdm(range(args.num_epoch)):
            train_loss = run_train(train_dataloader, net, optimizer, criterion)

            with torch.no_grad():
                acc, valid_loss = run_valid(valid_dataloader, net, criterion)

            if writer:
                writer.add_scalars('Learning curve', {
                    'Train loss': train_loss,
                    'Validation loss': valid_loss,
                    'Validation accuracy': acc
                }, i)

            if (i+1) % 10 == 0:
                torch.save({
                    'epoch': i,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, './outputs/checkpoint-{}.tar'.format(i+1))

        # save the model at the end of training
        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, './outputs/complete.tar')

        with torch.no_grad():
            acc, test_loss = run_test(test_dataloader, net, criterion)

    print('Test accuracy: {}%'.format(acc))
    print('Test loss:', test_loss)


if __name__ == '__main__':
    main()
