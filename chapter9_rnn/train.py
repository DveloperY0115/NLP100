"""
train.py: Training routine for RNN models
"""

import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import BucketIterator
import utils.fields as fields

from models import SimpleRNN, SimpleLSTM, SimpleGRU

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    type=str,
    default="gru",
    help="Type of model to use. Can be one of 'rnn', 'lstm', 'gru'.",
)
parser.add_argument("--num_epoch", type=int, default=100, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=10, help="Size of a batch")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--beta1", type=float, default=0.9, help="Beta 1 of Adam optimizer")
parser.add_argument("--beta2", type=float, default=0.999, help="Beta 2 of Adam optimizer")
parser.add_argument("--step_size", type=int, default=100, help="Step size of StepLR")
parser.add_argument("--gamma", type=float, default=0.99, help="Gamma of StepLR")
args = parser.parse_args()


def main():

    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[!] Using {}".format(device))

    # load train, valid, test data
    train_data, valid_data, test_data = TabularDataset.splits(
        path="./data/",
        train="train.csv",
        validation="valid.csv",
        test="test.csv",
        format="csv",
        fields=[
            ("title", fields.TITLE),
            ("publisher", fields.PUBLISHER),
            ("category", fields.CATEGORY),
        ],
        skip_header=True,
        csv_reader_params={"delimiter": ","},
    )

    # build vocabulary
    fields.TITLE.build_vocab(train_data, min_freq=5, max_size=10000)
    fields.TITLE.build_vocab(valid_data, min_freq=5, max_size=10000)
    fields.TITLE.build_vocab(test_data, min_freq=5, max_size=10000)
    fields.PUBLISHER.build_vocab(train_data, valid_data, test_data)
    fields.CATEGORY.build_vocab(train_data, valid_data, test_data)

    vocab_size = len(fields.TITLE.vocab)

    # create data loader for each dataset
    train_loader = BucketIterator(dataset=train_data, batch_size=args.batch_size, device=device)
    valid_loader = BucketIterator(dataset=valid_data, batch_size=args.batch_size, device=device)
    test_loader = BucketIterator(dataset=test_data, batch_size=args.batch_size, device=device)

    # create model
    if args.model_type == "rnn":
        model = SimpleRNN(2, 32, vocab_size, 128, 5, num_publisher=6).to(device)
    elif args.model_type == "lstm":
        model = SimpleLSTM(2, 32, vocab_size, 128, 5, num_publisher=6).to(device)
    elif args.model_type == "gru":
        model = SimpleGRU(2, 32, vocab_size, 128, 5, num_publisher=6).to(device)
    else:
        print("[!] Invalid configuration. Please choose one of 'rnn', 'lstm', 'gru'.")
        return -1

    # configure optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # iterate over epochs
    for epoch in tqdm(range(args.num_epoch)):

        train_iter = iter(train_loader)

        while True:
            # iterate over the dataset
            try:
                train_batch = next(train_iter)
            except StopIteration:
                break

            title = train_batch.title.to(device)
            publisher = train_batch.publisher.to(device)
            category = train_batch.category.to(device)

            # one-hot encoding for categorical data
            publisher = F.one_hot(publisher, 6)
            category = F.one_hot(category, 5)

            # forward prop
            pred = model(title, publisher)

            # calculate loss
            loss = nn.CrossEntropyLoss()(pred, torch.argmax(category, dim=1))

            # back prop
            loss.backward()
            optimizer.step()

            # update scheduler
            scheduler.step()

        with torch.no_grad():

            test_iter = iter(test_loader)
            test_batch = next(test_iter)

            """
            evaluate, record
            """


if __name__ == "__main__":
    main()
