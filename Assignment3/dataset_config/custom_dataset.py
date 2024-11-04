from torch.utils.data import Dataset
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = self.load_data()

    @staticmethod
    def load_data():
        data = [("sample_image_path.jpg", 0) for _ in range(100)]
        return data

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = Image.open(os.path.join(self.root, img_path))

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)
