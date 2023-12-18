import argparse
import os

import torch
from scripts.dataset import CELEBA_EASY_LABELS, setup_data_loaders
from scripts.model import CCVAE
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from scripts.utils import Files
import torch.distributions as dist

def main(arguments):
    
    im_shape = (3, 64, 64)
    device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    files = Files(arguments.datasets_path)
    print('Loading and Splitting Dataset')

    model = CCVAE(
        z_dim=arguments.z_dim,
        y_prior_params=0.5 * torch.ones(len(CELEBA_EASY_LABELS), device=device),
        num_classes=len(CELEBA_EASY_LABELS),
        device=device,
        image_shape=im_shape
        )

    model.load(arguments.model_path)
    model.to(device)
    model.eval()
    data_loaders = setup_data_loaders(
        files.datasets_folder,
        arguments.supervised_fraction,
        batch_size=arguments.batch_size,
        validation_size=arguments.validation_size)

    image, label = next(iter(data_loaders['test']))
    image = image.to(device=device)
    label = label.to(device=device)
    # EXPERIMENTS
    image_indices = [0, 80, 52, 90, 89, 63, 50, 87]
    
    # CONDITIONAL GENERATION
    image_index = 0
    number_of_samples = 4
    for image_index in image_indices:
        image_name = data_loaders['test'].dataset.filename[image_index]
        reconstucted_samples = model.conditional_generation(image[image_index, :], label[image_index, :], num_sample=number_of_samples)
        save_image(
            image[image_index, :],
            os.path.join(files.output_folder,
            f'original_'+image_name))

        save_image(
            make_grid(reconstucted_samples, nrow=number_of_samples),
            os.path.join(files.output_folder,
            f'test_conditional_generation_'+image_name))

    # VARIANCE INTERVENING
    for image_index in image_indices:
        image_name = data_loaders['test'].dataset.filename[image_index]
        y = label[image_index, :]
        labels_intervation = image[image_index, :].view(1, im_shape[0], im_shape[1], im_shape[2])
        for i, label_name in enumerate(['Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses', 'Male','Young']):
            if y[i] == 1:
                r = torch.zeros_like(image[image_index, :], device=device).view(1, im_shape[0], im_shape[1], im_shape[2])
            else:
                y[i] = 1
                r = model.conditional_generation(
                    image[image_index, :],
                    y,
                    num_sample=1)
                y[i] = 0
            labels_intervation = torch.cat((labels_intervation, r), axis=0)
        transform = transforms.Grayscale()
        dif = torch.round(torch.abs(transform(labels_intervation) - transform(image[image_index, :])))
        save_image(
            make_grid(dif, nrow=len(image_indices)),
            os.path.join(files.output_folder,
            f'test_intervation_diff_on_label_{label_name}_'+image_name))
        save_image(
                make_grid(labels_intervation, nrow=len(image_indices)),
                os.path.join(files.output_folder,
                f'test_intervation_on_label_{label_name}_'+image_name))

    # CHARACTERISTIC SWAP
    for i in image_indices:
        image_name_1 = data_loaders['test'].dataset.filename[i]
        image_1 = image[i, :]
        row_recon = image_1.view(1, im_shape[0], im_shape[1], im_shape[2])
        for j in image_indices:
            image_name_2 = data_loaders['test'].dataset.filename[j]
            image_2 = image[j, :]
            z1 = dist.Normal(*model.encoder(image_1)).sample()
            zc1, zs1 = z1.split([model.num_labeled, model.num_unlabeled], 1)
            z2 = dist.Normal(*model.encoder(image_2)).sample()
            zc2, zs2 = z2.split([model.num_labeled, model.num_unlabeled], 1)
            new_z_1 = torch.cat((zc1, zs2), axis=1)
            r1 = model.decoder(new_z_1)
            row_recon = torch.cat((row_recon, r1), axis=0)

        save_image(
            make_grid(row_recon, nrow=len(image_indices)+1),
            os.path.join(files.output_folder,
            f'test_swap_zc1_zs2_{image_name_2[:-4]}_'+image_name_1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        default='data/output',
                        type=str,
                        help='Model path'
                        )
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
    parser.add_argument('--validation_size',
                        default=20000,
                        type=int,
                        help='Validation total number of samples'
                        )
    args = parser.parse_args()

    main(args)
