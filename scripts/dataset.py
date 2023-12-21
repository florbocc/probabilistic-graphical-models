import os
from typing import Any, Tuple
import PIL
import torch
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


CELEBA_LABELS = ['5_o_Clock_Shadow', 'Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair','Bushy_Eyebrows', \
                 'Chubby', 'Double_Chin','Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open','Mustache','Narrow_Eyes', 'No_Beard', 'Oval_Face', \
                 'Pale_Skin','Pointy_Nose','Receding_Hairline','Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', \
                 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

CELEBA_EASY_LABELS = ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bangs', 'Black_Hair', 'Blond_Hair','Brown_Hair','Bushy_Eyebrows', 'Chubby','Eyeglasses', 'Heavy_Makeup', 'Male', \
                      'No_Beard', 'Pale_Skin', 'Receding_Hairline', 'Smiling', 'Wavy_Hair', 'Wearing_Necktie', 'Young']


class ModifiedCelebA(CelebA):
    valid_labels_index = [i for i, l in enumerate(CELEBA_LABELS) if l in CELEBA_EASY_LABELS]
    _prior_params = torch.ones(1, len(CELEBA_EASY_LABELS)) / 2

    # Class to only use easy labels
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        target = self.attr[index].float()
        imagepath = os.path.join(
            self.root,
            self.base_folder,
            "img_align_celeba",
            self.filename[index])
        X = self.transform(PIL.Image.open(imagepath))
        target = target[self.valid_labels_index]
        return X, target

    def update_prior(self):
        self._prior_params = torch.mean(
            self.attr[:, self.valid_labels_index].float(),
            dim=0)

    def labels_prior_params(self):
        return self._prior_params


def setup_data_loaders(
        dataset_folderbase: str,
        supervised_fraction: float,
        batch_size: int,
        validation_size: int,
        *args,
        **kwargs) -> None:
    transform = transforms.Compose([
                                transforms.Resize((64, 64)),
                                transforms.ToTensor()
                            ])
    test_data = ModifiedCelebA(
        dataset_folderbase,
        split='test',
        transform=transform,
        *args,
        **kwargs)
    train_data = ModifiedCelebA(
        dataset_folderbase,
        split='train',
        transform=transform,
        *args,
        **kwargs)

    # FIXME: This could be improved :)
    # The class CelebA do not have a copy method :(
    # The following lines are a go around to avoid creating
    # a new class.
    supervised_train_data = ModifiedCelebA(
        dataset_folderbase,
        split='train',
        transform=transform,
        *args,
        **kwargs)
    unsupervised_train_data = ModifiedCelebA(
        dataset_folderbase,
        split='train',
        transform=transform,
        *args,
        **kwargs)
    validation_data = ModifiedCelebA(
        dataset_folderbase,
        split='valid',
        transform=transform,
        *args,
        **kwargs)
    # rewrite the attributes and filenames
    validation_data.attr = train_data.attr[-validation_size:]
    validation_data.filename = train_data.filename[-validation_size:]

    X = train_data.filename[:-validation_size]
    y = train_data.attr[:-validation_size]
    # FIXME: we are not considering the case sf= 0 o 1
    if supervised_fraction > 0 and supervised_fraction < 1:
        split = int(supervised_fraction * len(X))
        supervised_train_data.attr = y[:split]
        supervised_train_data.filename = X[:split]
        supervised_train_data.update_prior()
        unsupervised_train_data.attr = y[split:]
        unsupervised_train_data.filename = X[split:]

        # loaders
        loaders = {
            'unsupervised': DataLoader(unsupervised_train_data, batch_size=batch_size, shuffle=True, **kwargs),
            'supervised': DataLoader(supervised_train_data, batch_size=batch_size, shuffle=True, **kwargs),
            'validation': DataLoader(validation_data, batch_size=batch_size, shuffle=True, **kwargs),
            'test': DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs),
        }
    elif supervised_fraction == 1:
        supervised_train_data.attr = y
        supervised_train_data.filename = X
        supervised_train_data.update_prior()
        loaders = {
            'unsupervised': None,
            'supervised': DataLoader(supervised_train_data, batch_size=batch_size, shuffle=True, **kwargs),
            'validation': DataLoader(validation_data, batch_size=batch_size, shuffle=True, **kwargs),
            'test': DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs),
        }
    elif supervised_fraction == 0:
        unsupervised_train_data.attr = y
        unsupervised_train_data.filename = X

        # loaders
        loaders = {
            'unsupervised': DataLoader(unsupervised_train_data, batch_size=batch_size, shuffle=True, **kwargs),
            'supervised': None,
            'validation': DataLoader(validation_data, batch_size=batch_size, shuffle=True, **kwargs),
            'test': DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs),
        }
    return loaders
