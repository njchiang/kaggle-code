import multiprocessing

from torch.utils.data import Dataset, DataLoader

class BaseDataSet(Dataset):
    def __init__(self):
        self._data = []
        self._labels = None
        self._with_labels = False

    def __len__(self):
        self._len = len(self._data)
        return self._len 

    def __getitem__(self, idx):
        if self._labels is not None:
            return self._data[idx], self._labels[idx]
        else: 
            return self._data[idx], None