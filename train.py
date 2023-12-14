import argparse
from dataclasses import dataclass

import torch
from tqdm import tqdm
from scripts.dataset import setup_data_loaders
from scripts.model import CCVAE
from scripts.utils import Files
from torch.utils.tensorboard import SummaryWriter


def main(arguments):
    writer = SummaryWriter()
    im_shape = (3, 64, 64)
    device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    files = Files(arguments.datasets_path)
    print('Loading and Splitting Dataset')
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
        n_supervised_batches = len(data_loaders['supervised'])
        Tsup = batches_per_epoch // n_supervised_batches

        model = CCVAE(z_dim=45,
                      y_prior_params=data_loaders['test'].dataset.labels_prior_params().to(device=device),
                      num_classes=18,
                      device=device,
                      image_shape=im_shape
                      )
        model.to(device)
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=arguments.learning_rate)
        count_sup = 0

        # In this case we want to train with both supervised
        # and unsupervised dataset. The first one is going to
        # be iterated at a period Tsup
        for i in tqdm(range(batches_per_epoch), desc='Batch per epoch'):
            if i % Tsup == 0 and count_sup < n_supervised_batches:
                (images, labels) = next(supervised_batch)
                loss = model.supervised_ELBO(
                    images.to(device),
                    labels.to(device))
                epoch_loss_supervised+=loss.detach().item()
                count_sup += 1
            else:
                (images, _) = next(unsupervised_batch)
                loss = model.unsupervised_ELBO(images.to(device))
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
        )
        writer.add_scalar('Loss/train_supervised', epoch_loss_supervised, epoch)
        writer.add_scalar('Loss/train_unsupervised', epoch_loss_unsupervised, epoch)
        writer.add_scalar('accuracy', validation_accuracy, epoch)
    writer.close()

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
    parser.add_argument('--learning_rate',
                        default=1e-4,
                        type=float,
                        help='Learning rate'
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
