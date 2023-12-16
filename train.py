import argparse
import os
import PIL

import torch
from torchvision.utils import save_image, make_grid
import torch.distributions as dist
from tqdm import tqdm
from scripts.dataset import setup_data_loaders, CELEBA_EASY_LABELS
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

    model = CCVAE(
        z_dim=arguments.z_dim,
        y_prior_params=data_loaders['supervised'].dataset.labels_prior_params().to(device=device),
        num_classes=len(CELEBA_EASY_LABELS),
        device=device,
        image_shape=im_shape
        )
    model.to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=arguments.learning_rate)
    if arguments.debug:
        # Save original image 
        debug_images_indices = [0, 50, 80, 90]
        for index in debug_images_indices:
            image_name = data_loaders['test'].dataset.filename[index]
            image_filepath = os.path.join(files.celeba_dataset_filepath, 'img_align_celeba', image_name)
            image = PIL.Image.open(image_filepath)
            image.save(os.path.join(files.output_folder, 'original_'+image_name))
    for epoch in range(arguments.max_epochs):
        epoch_loss_supervised = 0
        epoch_loss_unsupervised = 0

        supervised_batch = iter(data_loaders['supervised'])
        unsupervised_batch = iter(data_loaders['unsupervised'])
        
        # FIXME: Again not considering limits 0 and 1
        n_supervised_batches = len(data_loaders['supervised'])
        batches_per_epoch = n_supervised_batches +\
            len(data_loaders['unsupervised'])
        Tsup = batches_per_epoch // n_supervised_batches
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
              f" validation accuracy {validation_accuracy:.2f}"
              )
        writer.add_scalar('Loss/train_supervised', epoch_loss_supervised, epoch)
        writer.add_scalar('Loss/train_unsupervised', epoch_loss_unsupervised, epoch)
        writer.add_scalar('accuracy', validation_accuracy, epoch)

        if arguments.debug:
            # Save reconstruction of each image
            transform = data_loaders['test'].dataset.transform
            for index in debug_images_indices:
                image_name = data_loaders['test'].dataset.filename[index]
                image_filepath = os.path.join(files.celeba_dataset_filepath, 'img_align_celeba', image_name)
                image = transform(PIL.Image.open(image_filepath))
                r = model.reconstruction(image.to(device=device))
                save_image(r, os.path.join(files.output_folder, f'epoch_{epoch}_'+image_name))
            
            # Latent walk at every step
            # Given a selected label save 3 samples per epoch
            selected_labels = ['Blond_Hair', 'Smiling', 'Wavy_Hair', 'Wearing_Necktie']
            index_labels = [i for i, l in enumerate(CELEBA_EASY_LABELS) if l in selected_labels]
            
            #Select an image 
            index = 0
            image_name = data_loaders['test'].dataset.filename[index]
            image_filepath = os.path.join(files.celeba_dataset_filepath, 'img_align_celeba', image_name)
            image = transform(PIL.Image.open(image_filepath))
            
            a = 8
            Ns = 5
            z_ = dist.Normal(*model.encoder(image)).sample()
            for index in index_labels:
                z = z_.clone()
                z = z.expand(Ns, -1).contiguous()
                y = torch.zeros(1, len(CELEBA_EASY_LABELS), device=device)
                mu_false, sigma_false = model.conditional_prior(y)
                y[:, index].fill_(1.0)
                mu_true, sigma_true = model.conditional_prior(y)
                sign = torch.sign(mu_true[:, index] - mu_false[:, index])
                z_false_lim = (mu_false[:, index] - a * sign * sigma_false[:, index]).item()    
                z_true_lim = (mu_true[:, index] + a * sign * sigma_true[:, index]).item()
                z[:, index] = torch.linspace(z_false_lim, z_true_lim, Ns)

                imgs = model.decoder(z).view(-1, *im_shape)
                grid = make_grid(imgs, nrow=Ns)
                save_image(grid, os.path.join(files.output_folder, f"epoch_{epoch}_latent_walk_{CELEBA_EASY_LABELS[index]}.png"))



    test_accuracy = model.accuracy(data_loaders['test'])
    print("Test acc %.3f" % test_accuracy)
    writer.close()
    model.save(files.output_folder)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_path',
                        default='data/datasets',
                        type=str,
                        help='Dataset path'
                        )
    parser.add_argument('--z_dim',
                        default=45,
                        type=int,
                        help='Latent space dimension'
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
                        default=35,
                        type=int,
                        help='Maximum number of epochs'
                        )
    parser.add_argument('--validation_size',
                        default=20000,
                        type=int,
                        help='Validation total number of samples'
                        )
    parser.add_argument('--debug',
                        action='store_true')
    args = parser.parse_args()

    main(args)
