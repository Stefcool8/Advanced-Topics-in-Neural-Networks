import importlib

from PIL import Image
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

from dataset_config.cached_dataset import CachedDataset


class DatasetFactory:
    @staticmethod
    def get_dataset(config, train=True, dataset_transforms=None):
        if dataset_transforms is None:
            dataset_transforms = []
        dataset_name = config["dataset"]["name"]
        dataset_path = config["dataset"]["path"]

        if dataset_name.lower() in ["mnist", "cifar10", "cifar100"]:
            return DatasetFactory.get_standard_dataset(dataset_name, dataset_path, train, dataset_transforms)

        dataset_class = DatasetFactory.load_class(config["dataset"]["class"])
        return dataset_class(train=train)

    @staticmethod
    def get_standard_dataset(name, path, train, dataset_transforms):
        """
        Loads standard torchvision datasets, applying each transform separately.
        """
        # Base dataset without transforms to preserve original images
        if name.lower() == "mnist":
            base_dataset = datasets.MNIST(root=path, train=train, transform=None, download=True)
        elif name.lower() == "cifar10":
            base_dataset = datasets.CIFAR10(root=path, train=train, transform=None, download=True)
        elif name.lower() == "cifar100":
            base_dataset = datasets.CIFAR100(root=path, train=train, transform=None, download=True)
        else:
            raise ValueError(f"Unsupported dataset: {name}")

        transformed_datasets = []

        original_data = [(transforms.ToTensor()(Image.fromarray(img)), label) for img, label in zip(base_dataset.data, base_dataset.targets)]
        transformed_datasets.append(CachedDataset(original_data))

        for transform_name in dataset_transforms:
            aug_transform = DatasetFactory.get_transform_pipeline(transform_name)
            transformed_data = [(aug_transform(Image.fromarray(img)), label) for img, label in zip(base_dataset.data, base_dataset.targets)]
            transformed_datasets.append(CachedDataset(transformed_data))

        # Concatenate all transformed datasets
        augmented_dataset = ConcatDataset(transformed_datasets)

        return augmented_dataset

    @staticmethod
    def get_transform_pipeline(name):
        """
        Returns a composed transform pipeline based on the name.
        """
        common_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]

        augmentation_transforms = {
            'RandomHorizontalFlip': transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), *common_transforms]),
            'RandomCrop': transforms.Compose([transforms.RandomResizedCrop(size=(32, 32), scale=(0.8, 1.0)), *common_transforms]),
            'RandomPerspective': transforms.Compose([transforms.RandomPerspective(distortion_scale=0.4, p=1.0), *common_transforms]),
            'ColorJitter': transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), *common_transforms]),
            'RandomErasing': transforms.Compose([transforms.ToTensor(), transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
            'AutoAugment': transforms.Compose([transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10), *common_transforms])
        }

        return augmentation_transforms.get(name, transforms.Compose(common_transforms))

    @staticmethod
    def load_class(class_path):
        """
        Dynamically loads a class from a string path.
        """
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
