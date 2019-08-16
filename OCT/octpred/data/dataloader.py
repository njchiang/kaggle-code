import os
import multiprocessing

import torch
import torchvision
from torchvision import datasets, models, transforms

MAX_CPU_COUNT = max(multiprocessing.cpu_count() // 2, 1)

data_dir = '~/Downloads/OCT2017'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

class OCTDataSet:
    def __init__(self, data_dir, batch_size=8, num_workers=None):
        self.data_dir = data_dir

        num_workers = num_workers if num_workers else MAX_CPU_COUNT
        data_transforms = self._create_transforms()
        image_datasets, class_names = self._create_folder_datasets(data_dir, data_transforms)
        dataloaders, dataset_sizes = self.create_dataloaders(image_datasets)

        for x in [TRAIN, VAL, TEST]:
            print("Loaded {} images under {}".format(dataset_sizes[x], x))

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
        ) for x in [TRAIN, VAL, TEST]}
        self._class_names = self._image_datasets[TRAIN].classes
        return self._image_datasets, self._class_names

    def create_dataloaders(self, image_datasets, batch_size=8, num_workers=MAX_CPU_COUNT):

        self._dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x], batch_size=batch_size,
                shuffle=True, num_workers=num_workers
            )
            for x in [TRAIN, VAL, TEST]}
        self._dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

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
