import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader


class DataPrefetcher(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

