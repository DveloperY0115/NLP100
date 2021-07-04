import torch
from torchtext.legacy import data

from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator
from torchtext.legacy.data.iterator import batch

import utils.fields as fields

def main():

    # check if GPU is available
    if torch.cuda.is_available():
        device='gpu'
        print('[!] CUDA device detected.')
    else:
        device='cpu'
        print('[!] No available CUDA device.')
    print('[!] Using {}'.format(device))
    
    # load train, valid, test data
    train_data, valid_data, test_data = TabularDataset.splits(
        path='./data/', train='train.csv', validation='valid.csv', test='test.csv', format='csv',
        fields=[('title', fields.TITLE), ('publisher', fields.PUBLISHER), ('category', fields.CATEGORY)],
        skip_header=True,
        csv_reader_params={'delimiter':','}
    )

    # build vocabulary
    fields.TITLE.build_vocab(train_data, min_freq=5, max_size=10000)
    fields.TITLE.build_vocab(valid_data, min_freq=5, max_size=10000)
    fields.TITLE.build_vocab(test_data, min_freq=5, max_size=10000)
    fields.PUBLISHER.build_vocab(train_data, valid_data, test_data)
    fields.CATEGORY.build_vocab(train_data, valid_data, test_data)

    # set training parameters
    batch_size = 10
    num_epoch = 100

    # create data loader for each dataset
    train_loader = Iterator(dataset=train_data, batch_size=batch_size, device=device)
    valid_loader = Iterator(dataset=valid_data, batch_size=batch_size, device=device)
    test_loader = Iterator(dataset=test_data, batch_size=batch_size, device=device)

    # iterate over epochs
    for epoch in range(num_epoch):

        train_iter = iter(train_loader)

        while True:
            # iterate over the dataset
            try:
                train_batch = next(train_iter)
            except StopIteration:
                break

            title = train_batch.title
            publisher = train_batch.publisher
            category = train_batch.category
            
            """
            some training -> prop, back prop, optimize
            """


        with torch.no_grad():

            test_iter = iter(test_loader)
            test_batch = next(test_iter)

            """
            evaluate, record
            """

        
if __name__ == '__main__':
    main()