from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
import PIL
import csv
import preprocessing


class AugmentedSignLanguageMNIST(Dataset):
    """
    MNIST Sign Language dataset
    Samples saved as Image (1 x 1 x 28 x 28) together with their Label
    """

    @staticmethod
    def map_to_alphabet():
        """
        Mapping for one-hot encoding
        :return:
        """
        # letters J and Z are excluded because they require motion
        # so only 24 valid labels in 0 to 25 alphabet
        # labels mapped to between 0 and 23 to make set contiguous
        alphabet_map = list(range(25))
        alphabet_map.remove(9)

        return alphabet_map

    @staticmethod
    def extract(path: str):
        """
        Extracts labels and images from the CSV file
        28 x 28 = 784 pixel values per sample

        """
        mapping = AugmentedSignLanguageMNIST.map_to_alphabet()
        labels = []
        images = []

        with open(path) as file:
            _ = next(file)
            reader = csv.reader(file)
            for sample in reader:
                # First value is label, followed by pixel values
                label = int(sample[0])
                labels.append(mapping.index(label))
                images.append(list(map(int, sample[1:])))
        return labels, images

    def __init__(self, path="data/sign_mnist_train.csv", mean=[0.485], std=[0.229]):
        """
        Args:
            path: Path to mnist training file
            mean & std: obtained from data exploration
        """
        labels, images = self.extract(path)

        self._mean = mean
        self._std = std

        self._images = np.array(images, dtype=np.uint8).reshape((-1, 28, 28, 1))
        self._labels = np.array(labels, dtype=np.uint8).reshape((-1, 1))

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),

            # data augmentation
            transforms.ColorJitter(hue=0.05, saturation=0.05),
            transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),

            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)])

        return {
            'image': transform(self._images[index]).float(),
            'label': torch.from_numpy(self._labels[index]).float()
        }


def AUGMENT_get_train_test_loader(batch_size=32):
    trainset = AugmentedSignLanguageMNIST('data/sign_mnist_train.csv')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return trainloader, trainset


def concat_get_train_test_loader(batch_size=32):
    augmented_train_loader, augmented_trainset = AUGMENT_get_train_test_loader()
    trainloader, testloader, trainset, _ = preprocessing.get_train_test_loader()
    concat_trainset = torch.utils.data.ConcatDataset([trainset, augmented_trainset])
    concat_trainloader = torch.utils.data.DataLoader(dataset=concat_trainset, batch_size=32, shuffle=True)

    return concat_trainloader, testloader
