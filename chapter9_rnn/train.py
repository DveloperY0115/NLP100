"""
train.py: Training routine for RNN models
"""

import os
import argparse
from torch.serialization import save
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from models import SimpleRNN, SimpleLSTM, SimpleGRU
from dataset import AggregatorNewsDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    type=str,
    default="lstm",
    help="Type of model to use. Can be one of 'rnn', 'lstm', 'gru'.",
)
parser.add_argument("--num_layer", type=int, default=3, help="Number of recurrent layers")
parser.add_argument(
    "--hidden_dim", type=int, default=128, help="Dimensionality of hidden & cell state"
)
parser.add_argument(
    "--embed_dim", type=int, default=256, help="Dimensionality of word embedding vector"
)
parser.add_argument("--num_epoch", type=int, default=300, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Size of a batch")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--beta1", type=float, default=0.9, help="Beta 1 of Adam optimizer")
parser.add_argument("--beta2", type=float, default=0.999, help="Beta 2 of Adam optimizer")
parser.add_argument("--step_size", type=int, default=10000, help="Step size of StepLR")
parser.add_argument("--gamma", type=float, default=0.8, help="Gamma of StepLR")
parser.add_argument(
    "--save_interval", type=int, default=100, help="Interval between each checkpoint"
)
args = parser.parse_args()


def main():

    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[!] Using {}".format(device))

    dataset = AggregatorNewsDataset("data/train.csv")
    train_data, test_data = data.random_split(dataset, [len(dataset) - 1000, 1000])

    print("[!] Cardinality of training data: {}".format(len(train_data)))
    print("[!] Cardinality of test data: {}".format(len(test_data)))

    # setup summary writer
    writer = SummaryWriter(os.path.join("out", args.model_type, "log"))

    # create data loader for each dataset
    train_loader = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)

    # create model
    if args.model_type == "rnn":
        model = SimpleRNN(
            args.num_layer,
            args.hidden_dim,
            dataset.vocab_size,
            args.embed_dim,
            5,
            0.7,
            num_publisher=5,
        ).to(device)
    elif args.model_type == "lstm":
        model = SimpleLSTM(
            args.num_layer,
            args.hidden_dim,
            dataset.vocab_size,
            args.embed_dim,
            5,
            0.7,
            num_publisher=5,
        ).to(device)
    elif args.model_type == "gru":
        model = SimpleGRU(
            args.num_layer,
            args.hidden_dim,
            dataset.vocab_size,
            args.embed_dim,
            5,
            0.7,
            num_publisher=5,
        ).to(device)
    else:
        print("[!] Invalid configuration. Please choose one of 'rnn', 'lstm', 'gru'.")
        return -1

    # configure optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # iterate over epochs
    for epoch in tqdm(range(args.num_epoch)):
        loss = train_one_epoch(model, optimizer, scheduler, train_loader, epoch, device)

        with torch.no_grad():
            test_loss, test_accuracy = test_one_epoch(model, device, test_loader, epoch)

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_accuracy, epoch)

        # save model & checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(epoch, loss, model, optimizer, scheduler)


def train_one_epoch(model, optimizer, scheduler, train_loader, epoch, device):
    train_iter = iter(train_loader)

    train_loss = 0
    train_idx = 0

    while True:
        # iterate over the dataset
        try:
            train_batch = next(train_iter)
        except StopIteration:
            break

        title, publisher, category = train_batch

        title = title.to(device)
        publisher = publisher.to(device)
        category = category.to(device)

        # initialize gradient
        optimizer.zero_grad()

        # forward prop
        pred = model(title, publisher)

        # calculate loss
        loss = nn.CrossEntropyLoss()(pred, torch.argmax(category, dim=1))

        # back prop
        loss.backward()
        optimizer.step()

        # update scheduler
        scheduler.step()

        # accumulate loss & record iteration
        train_loss += loss.item()
        train_idx += 1

    # average train loss
    train_loss /= train_idx

    print("------------------------------------------")
    print("[!] Epoch {} train loss: {}".format(epoch, train_loss))
    print("------------------------------------------")

    return train_loss


def test_one_epoch(model, device, test_loader, epoch):
    test_iter = iter(test_loader)
    test_batch = next(test_iter)

    title_test, publisher_test, category_test = test_batch

    title_test = title_test.to(device)
    publisher_test = publisher_test.to(device)
    category_test = category_test.to(device)

    pred_test = model(title_test, publisher_test)

    test_loss = nn.CrossEntropyLoss()(pred_test, torch.argmax(category_test, dim=1))

    print("------------------------------------------")
    print("[!] Epoch {} test loss: {}".format(epoch, test_loss.item()))
    print("------------------------------------------")

    test_accuracy = (torch.argmax(pred_test, dim=1) == torch.argmax(category_test, dim=1)).int()
    test_accuracy = torch.sum(test_accuracy) / pred_test.size()[0] * 100

    print("------------------------------------------")
    print("[!] Epoch {} test accuracy: {}%".format(epoch, test_accuracy))
    print("------------------------------------------")

    return test_loss, test_accuracy


def save_checkpoint(epoch, loss, model, optimizer, scheduler):

    if not os.path.exists("out"):
        os.mkdir("out/")

    save_root = "out/{}/".format(args.model_type)

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    save_path = os.path.join(save_root, "checkpoint-epoch-{}.pt".format(epoch + 1))

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        save_path,
    )
    print("[!] Saved model at: {}".format(save_path))


if __name__ == "__main__":
    main()
