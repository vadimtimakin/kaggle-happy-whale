import enum
import json
import os
import random
import cv2

import albumentations as A
import numpy as np
import pandas as pd
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm
from PIL import Image
from skimage import color
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold

from objects.transforms import SmartResize


class DolphinDataset(Dataset):
    """Dataset structure."""

    def __init__(self, config, images, labels, is_train, epoch):
        super().__init__()

        self.is_train = is_train
        self.epoch = epoch

        self.labels, self.img_paths = [], []
        for img_name, text in zip(images, labels):
            self.img_paths.append(os.path.join(config.paths.path_to_images, img_name))
            self.labels.append(text)

        if config.training.debug:
            if is_train:
                self.labels = self.labels[:config.training.number_of_debug_train_samples]
            else:
                self.labels = self.labels[:config.training.number_of_debug_val_samples]

        self.data_len = len(self.labels)

        if is_train:
            self.transforms = A.Compose([
                A.Resize(384, 512, p=1.0),

                A.OneOf([
                    A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.1, p=0.75),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15, p=0.75),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
                ], p=0.75),

                A.OneOf([
                    A.Blur(blur_limit=2.5, p=0.75),
                    A.MotionBlur(blur_limit=(3, 5), p=0.75),
                ], p=0.75),

                A.OneOf([
                    A.Affine(shear=21, mode=cv2.BORDER_REPLICATE, p=0.75),
                    A.Compose([
                        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                        rotate_limit=16, border_mode=cv2.BORDER_REPLICATE, p=0.75),
                        A.GridDistortion(distort_limit=0.1, border_mode=cv2.BORDER_REPLICATE, p=0.75),
                    ], p=0.75)
                ], p=0.75),

                A.OneOf([
                    A.Cutout(num_holes=12, max_h_size=21, max_w_size=21, p=0.75),
                    A.Cutout(num_holes=6, max_h_size=42, max_w_size=42, p=0.75),
                ], p=0.75),
                
                A.CLAHE(clip_limit=2.5, p=0.75),
                A.ImageCompression(quality_lower=90, quality_upper=100, p=0.75),
                A.Normalize(p=1.0),
            ])
            
        else:
            self.transforms = A.Compose([
                A.Resize(384, 512, p=1.0),
                A.Normalize(p=1.0),
            ])

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        image = np.array(Image.open(img_path))
        if len(image.shape) != 3:
            image = np.stack((image, ) * 3, axis=-1)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        image = torch.from_numpy(image).unsqueeze(0)

        return image, label


def preprocess_dataframe(config, df) -> pd.DataFrame():
    """Preprocesses source dataframe."""

    # Species fix
    df.species.replace({"globis": "short_finned_pilot_whale",
                          "pilot_whale": "short_finned_pilot_whale",
                          "kiler_whale": "killer_whale",
                          "bottlenose_dolpin": "bottlenose_dolphin"}, inplace=True)

    # One hot encoding
    m = {}
    for i, u in enumerate(df[config.data.target_column].unique()):
        m[u] = i
    df[config.data.target_column] = df[config.data.target_column].map(m)

    # Creating stratify criterion
    df['species_and_ids'] = df['species'] + df['individual_id'].astype(str)

    return df


def oversampling(df: pd.DataFrame, target_column: str, oversampling_type: str, 
                 seed: int, downsample_higher_values: bool,
                 class_value: str, value: int) -> pd.DataFrame:
    """
    Oversamples samples for some classes in the dataframe
    up to the certain value.

    Args:

        df (pd.DataFrame): pandas dataframe to oversample.

        target_column (str): column with the labels
            which oversampling will be based on.

        oversampling_type (str): one of three types of oversampling.
        
            1) up_to_max - all the samples of each class
                will be oversampled up to the number of the
                most frequent class.

            2) by_class_value - all the samples of each class
                will be oversampled up to the number of the
                certain class. The name of this class has to be
                passed in the class_value parameter.

            3) by_value - all the samples of each class
                will be oversampled up to the custom number.
                This value has to be passed in the value parameter.

        class_value (str): used with the oversampling_type="by_class_value".
            class which frequency will be used for oversampling.

        value (int): used with the oversampling_type="by_value".
            value which all the classes will be upsampled to.

        seed (int): random seed.

        downsampling_higher_values (bool):
            If set to True, in case if oversampling_type is set to "by_class_value"
            or "by_value" and class_value or value parameter isn't equal
            to the number of samples of the most frequent class,
            the classes with the greater number of samples will be
            downsampled to this value.

    Returns:
        df (pd.DataFrame): the return oversampled dataframe.
    """

    columns = df.columns
    classes = df[target_column].unique()
    counts = df[target_column].value_counts()
    new_df = pd.DataFrame(columns=columns)

    # UP TO MAX
    if oversampling_type == "up_to_max":
        number = np.max(counts)

        for c in classes:
            one_class_df = df[df[target_column] == c]

            samples = one_class_df.sample(n=counts[c], random_state=seed).reset_index(drop=True)
            for _ in range(number // counts[c]):
                new_df = pd.concat([new_df, samples], ignore_index=True) 
            rest_samples = one_class_df.sample(n=number % counts[c],
                                               random_state=seed).reset_index(drop=True)
            new_df = pd.concat([new_df, rest_samples], ignore_index=True)

    # BY CLASS VALUE or BY CUSTOM VALUE
    else:
        number = counts[class_value] if oversampling_type == "by_class_value" else value

        for c in tqdm(classes):
            one_class_df = df[df[target_column] == c]

            if counts[c] > number:
                if downsample_higher_values == True:
                    samples = one_class_df.sample(n=counts[c], random_state=seed).reset_index(drop=True)
                    new_df = pd.concat([new_df, samples], ignore_index=True)
                else:
                    samples = df[df[target_column] == c]
                    new_df = pd.concat([new_df, samples], ignore_index=True)

            else:      
                samples = one_class_df.sample(n=counts[c], random_state=seed).reset_index(drop=True)
                for _ in range(number // counts[c]):
                    new_df = pd.concat([new_df, samples], ignore_index=True) 
                rest_samples = one_class_df.sample(n=number % counts[c],
                                                random_state=seed).reset_index(drop=True)
                new_df = pd.concat([new_df, rest_samples], ignore_index=True)

    # Shuffling
    new_df = new_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return new_df 


def get_fold_margins(config, current_fold):

    df = pd.read_csv(config.paths.path_to_csv)
    df = preprocess_dataframe(config, df)
    df = oversampling(df, 'individual_id', 'by_value', 0xFACED, False, None, 2)

    kfold = StratifiedKFold(n_splits=config.data.n_folds, shuffle=True, random_state=config.seed)
        
    for fold, (train_index, val_index) in enumerate(kfold.split(df, df[config.data.split_column])):
        if fold == current_fold:
            df = df.iloc[train_index]
            break

    df = df.drop_duplicates()

    tmp = np.sqrt(1 / np.sqrt(df['individual_id'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05
    return margins


def get_fold_samples(config, current_fold):

    df = pd.read_csv(config.paths.path_to_csv)
    df = preprocess_dataframe(config, df)

    kfold = StratifiedKFold(n_splits=config.data.n_folds, shuffle=True, random_state=config.seed)
        
    for fold, (train_index, val_index) in enumerate(kfold.split(df, df[config.data.split_column])):
        if fold == current_fold:
            train_images = df[config.data.id_column].iloc[train_index].values
            train_targets = df[config.data.target_column].iloc[train_index].values
            val_images = df[config.data.id_column].iloc[val_index].values
            val_targets = df[config.data.target_column].iloc[val_index].values
            break

    return train_images, train_targets, val_images, val_targets



def get_data_loader(config, is_train, images, labels, epoch):
    """Gets a PyTorch Dataloader."""
    dataset = DolphinDataset(config, images, labels, is_train, epoch)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=is_train,
        **config.data.dataloader_params,
    )
    return data_loader


def get_loaders(config, epoch, fold, is_train):
    """Get PyTorch Dataloaders."""

    train_images, train_targets, val_images, val_targets = get_fold_samples(config, fold)

    train_loader = get_data_loader(
        is_train=True * is_train,
        config=config,
        images=train_images,
        labels=train_targets,
        epoch=epoch,
    )

    val_loader = get_data_loader(
        is_train=False,
        config=config,
        images=val_images,
        labels=val_targets,
        epoch=epoch,
    )

    return train_loader, val_loader