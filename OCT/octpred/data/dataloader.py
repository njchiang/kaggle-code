import os
import multiprocessing
from pathlib import Path

from skimage import io
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, models, transforms, utils

MAX_CPU_COUNT = max(multiprocessing.cpu_count() // 2, 1)

data_dir = '~/Downloads/OCT2017'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
DS_NAME = 'amish'  # kermany


class OCTDataSet:
    def __init__(self, data_dir=None, batch_size=32, num_workers=None, ds_name="kermany", files_list=None, csv_file=None, pathology=None, num_val=1):
        assert ds_name in ["amish", "kermany"], "Unrecognized dataset name"
        
        num_workers = num_workers if num_workers else MAX_CPU_COUNT
        
        data_transforms = self._create_transforms()
        if ds_name.lower() == "kermany":
            assert data_dir, "No data directory specified"
            self.data_dir = data_dir
            self.partitions = [TRAIN, VAL, TEST]
            image_datasets, class_names = self._create_folder_datasets(data_dir, data_transforms)
        elif ds_name.lower() == "amish":
                    # files_list, csv_file,pathology, transform=None

            assert files_list, "Files list missing"
            assert csv_file, "Label csv missing"
            assert pathology, "Pathology (column name) missing"
            # need to split files_list into multiple files here
            n_files = len(files_list)
            train_n = n_files // 2
            val_n = train_n + num_val  # hope this works
            files_list = {
                TEST: files_list
            }
            self.partitions = [TEST]
            # files_list = {
            #     TRAIN: files_list[:train_n],
            #     VAL: files_list[train_n:val_n],
            #     TEST: files_list[val_n:]
            # }
            image_datasets, class_names = self._create_amish_datasets(files_list, csv_file, pathology, data_transforms)

        dataloaders, dataset_sizes = self.create_dataloaders(image_datasets, batch_size)

        for x in self.partitions:
            print("Loaded {} images under {}".format(dataset_sizes[x], x))
        
        if class_names:
            print("Classes: ")
            print(class_names)

    def _create_transforms(self):
        self._data_transforms = {
            TRAIN: transforms.Compose([
                # Data augmentation is a good practice for the train set
                # Here, we randomly crop the image to 224x224 and
                # randomly flip it horizontally. 
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            VAL: transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]),
            TEST: transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        }
        return self._data_transforms

    def _create_folder_datasets(self, data_dir, data_transforms=dict()):
        self._image_datasets = {    
            x: datasets.ImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms.get(x, transforms.ToTensor())
        ) for x in self.partitions}
        self._class_names = self._image_datasets[self.partitions[0]].classes
        return self._image_datasets, self._class_names

    def _create_amish_datasets(self, files_list, csv_file, pathology, data_transforms=dict()):
        # TODO IN PROGRESS. GET AMISHDATASET TO WORK PROPERLY
        # This will load the whole dataset for train, test, and val
        self._image_datasets = {
            # get labels and classes
            x: AmishDataset(files_list[x], csv_file, pathology, transform=data_transforms[x], mode=x)
        for x in self.partitions} # for x in [TRAIN, VAL, TEST]}
        # TODO figure out how to add these
        self._class_names = None
        return self._image_datasets, self._class_names

    def create_dataloaders(self, image_datasets, batch_size, num_workers=MAX_CPU_COUNT):

        self._dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], batch_size=batch_size,
                shuffle=True, num_workers=num_workers
            )
            for x in self.partitions}
        self._dataset_sizes = {x: len(image_datasets[x]) for x in self.partitions}

        return self._dataloaders, self._dataset_sizes

    def get_class_names(self):
        return self._class_names

    def get_transforms(self):
        return self._data_transforms

    def get_dataset_sizes(self):
        return self._dataset_sizes

    def get_dataloaders(self):
        return self._dataloaders

    def get_datasets(self):
        return self._image_datasets


# N.R. - loads a single pathology (column) in the csv file
class AmishDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, files_list, csv_file, pathology, transform=None, mode="train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            files_list (list): list of all imgs
            pathology (string): desired pathology
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file,index_col='PAT_ID')
        if mode == "test":  # make sure they are in the csv file for training mode
            included = files_list
            excluded = []
        else:
            included, excluded = self.checkPatIDs(files_list)

        self.files_list = np.array(included)
        self.excluded = np.array(excluded)
        self.pathology = pathology
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files_list[idx]
        
        # 3 steps: 1. read image, 2. copy 3x, 3. convert to pillow image
        image = io.imread(img_name)
        image = np.stack(3 * [image], 2)  # convert to RGB but only copies (following kermany)
        image = Image.fromarray(image)
        
        pat_id,eye = self.getPatID(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        try:
            pheno = self.df.loc[pat_id,f'{self.pathology}_{eye}']
        except KeyError:
            pheno = -1  # TODO : make this more robust
        # sample = {'image': image, 'pheno': pheno}
        
        # return sample
        return image, pheno
    
    def getPatID(self,img_name):
        img_name = Path(img_name)       
        return '_'.join(img_name.name.split('_')[:3]), img_name.name.split('_')[3]

    def checkPatIDs(self, files_list):
        included, excluded = [], []
        for img_name in files_list:
            pat_id, _ = self.getPatID(img_name)
            if pat_id in self.df.index:
                included.append(img_name)
            else:
                excluded.append(img_name)
        return included, excluded