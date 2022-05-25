from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch
from PIL import Image
import PIL
import csv

from sklearn.compose import TransformedTargetRegressor
import preprocessing
import pandas as pd
import matplotlib.pyplot as plt



class augment_SignLanguageMNIST(Dataset):
    """
    Sign Language Letters classification dataset.
    Each sample is 1 x 1 x 28 x 28 
    """

    @staticmethod
    def map_to_alphabet():
        # letters J and Z are exlcluded because they require motion 
        # so the dataset has labels 0-23 
        # this is transformed to alphabet labels 0-25 (26 letters)
        mapp = list(range(25))
        mapp.pop(9)

        return mapp

    @staticmethod
    def extract(path: str):
        """
        Extracts labels and samples from the CSV file
        28 x 28 = 784 pixel values per sample 

        """
        mapping = augment_SignLanguageMNIST.map_to_alphabet()
        labels = []
        samples = []

        with open(path) as f:
            _ = next(f)
            for line in csv.reader(f):
                label = int(line[0])
                labels.append(mapping.index(label))
                samples.append(list(map(int, line[1:])))
        return labels, samples

    def __init__(self, path="data/sign_mnist_train.csv", mean=[0.485], std=[0.229]):
        """
        Args:
            path: Path to mnist training file
        """
        labels, samples = self.extract(path)

        self._samples = np.array(samples, dtype=np.uint8).reshape((-1, 28, 28, 1))
        self._labels = np.array(labels, dtype=np.uint8).reshape((-1, 1))

        self._mean = mean
        self._std = std

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),

            #data augemnetation 
            transforms.ColorJitter(hue=0.05, saturation=0.05),
            transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),

            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)])

        return {
            'image': transform(self._samples[idx]).float(),
            'label': torch.from_numpy(self._labels[idx]).float()
        }


def AUGMENT_get_train_test_loader(batch_size=32):
    trainset = augment_SignLanguageMNIST('data/sign_mnist_train.csv')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    return trainloader, trainset


def concat_get_train_test_loader(batch_size=32):
    augmented_train_loader, augmented_trainset = AUGMENT_get_train_test_loader()
    trainloader, testloader , trainset, _  = preprocessing.get_train_test_loader()
    concat_trainset = torch.utils.data.ConcatDataset([trainset, augmented_trainset])
    concat_trainloader = torch.utils.data.DataLoader(dataset=concat_trainset, batch_size = 32, shuffle=True)

    return concat_trainloader, testloader



augmented_train_loader, augmented_trainset = AUGMENT_get_train_test_loader()
trainloader, _ , trainset, _  = preprocessing.get_train_test_loader()
print(len(trainloader))
print(len(augmented_train_loader))

concat_trainset = torch.utils.data.ConcatDataset([trainset, augmented_trainset])
concat_trainloader = torch.utils.data.DataLoader(dataset=concat_trainset, batch_size = 32, shuffle=True)

print(len(concat_trainloader))

it = iter(concat_trainloader)
train = next(it)

train_features = train['image']
train_labels = train['label']

# print(len(train_features[0]))


print(f"Feature batch shape: {train_features[0].size()}")
print(f"Labels batch shape: {train_labels.size()}")

fig, ax = plt.subplots(2, 5)

k = 0
for i in range(2) :
    for j in range(5):

        img = train_features[k].squeeze()
        label = train_labels[k]
        ax[i, j].imshow(img, cmap="gray")
        ax[i,j].set_title(str(label))
        print(f"Label: {label}")
        k = k + 1

plt.show()
