import os
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    def __init__(self, dataset, cache_dir="./cache", use_cache=True):
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache = [None] * len(self.dataset)

    def __getitem__(self, index):
        if self.use_cache and self.cache[index] is not None:
            return self.cache[index]

        data, target = self.dataset[index]
        if self.use_cache:
            self.cache[index] = (data.clone(), target)

        return data, target

    def __len__(self):
        return len(self.dataset)

    def clear_cache(self):
        """
        Clears the in-memory cache.
        """
        self.cache = [None] * len(self.dataset)
