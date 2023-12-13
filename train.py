import argparse
from dataclasses import dataclass

import torch
from tqdm import tqdm
from scripts.dataset import setup_data_loaders
from scripts.utils import Files


def main(arguments):
    im_shape = (3, 64, 64)
    device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    files = Files(arguments.datasets_path)
    data_loaders = setup_data_loaders(
        files.datasets_folder,
        arguments.supervised_fraction,
        batch_size=arguments.batch_size,
        validation_size=arguments.validation_size)

    for epoch in range(arguments.max_epochs):
        epoch_loss_supervised = 0
        epoch_loss_unsupervised = 0

        supervised_batch = iter(data_loaders['supervised'])
        unsupervised_batch = iter(data_loaders['unsupervised'])
        # FIXME: Again not considering limits 0 and 1
        batches_per_epoch = len(data_loaders['supervised']) +\
            len(data_loaders['unsupervised'])
        Tsup = batches_per_epoch // len(data_loaders['supervised'])

        model = CCVAE()
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=arguments.learning_rate)

        # In this case we want to train with both supervised  
        # and unsupervised dataset. The first one is going to
        # be iterated at a period Tsup
        for i in tqdm(range(batches_per_epoch)):
            if i % Tsup == 0:
                (images, labels) = next(supervised_batch)
                model.supervised(images, labels)
                loss = model.loss()
                epoch_loss_supervised+=loss.detach().item()
            else:
                (images, labels) = next(unsupervised_batch)
                model.unsupervised(images, labels)
                loss = model.loss()
                epoch_loss_unsupervised+=loss.detach().item()
            
            # backward
            loss.backward()
            # optimization step
            optimizer.step()
            optimizer.zero_grad()
            # validation
            with torch.no_grad():
                validation_accuracy = model.accuracy(
                    data_loaders['validation'])
        print(f"[Epoch {epoch}] Sup Loss "
              f"{epoch_loss_supervised:.3f},"
              f" Unsup Loss {epoch_loss_unsupervised:.3f},"
              f" Val Acc {validation_accuracy:.2f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path',
                        default='data/datasets',
                        type=str,
                        help='Dataset path'
                        )
    parser.add_argument('--supervised_fraction',
                        default=0.2,
                        type=float,
                        help='Fraction of supervised data'
                        )
    parser.add_argument('--batch_size',
                        default=200,
                        type=int,
                        help='Batch size'
                        )
    
    parser.add_argument('--max_epochs',
                        default=200,
                        type=int,
                        help='Maximum number of epochs'
                        )
    parser.add_argument('--validation_size',
                        default=20000,
                        type=int,
                        help='Validation total number of samples'
                        )
    args = parser.parse_args()

    main(args)
