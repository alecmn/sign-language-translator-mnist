import os

import cv2
import tqdm as tqdm
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch
import skimage
from sklearn.model_selection import train_test_split


class SignLanguageASL(Dataset):
    """
    Sign Language Letters classification dataset.
    Each sample is 1 x 1 x 28 x 28
    """

    @staticmethod
    def map_to_alphabet():
        # letters J and Z are excluded because they require motion
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
        mapping = SignLanguageASL.map_to_alphabet()
        labels = []
        samples = []

        for folderName in os.listdir(path):
            # if folderName not in [sign]:
            #     continue
            if not folderName.startswith('.'):
                if folderName in ['A']:
                    label = 0
                elif folderName in ['B']:
                    label = 1
                elif folderName in ['C']:
                    label = 2
                elif folderName in ['D']:
                    label = 3
                elif folderName in ['E']:
                    label = 4
                elif folderName in ['F']:
                    label = 5
                elif folderName in ['G']:
                    label = 6
                elif folderName in ['H']:
                    label = 7
                elif folderName in ['I']:
                    label = 8
                elif folderName in ['J']:
                    continue
                elif folderName in ['K']:
                    label = 10
                elif folderName in ['L']:
                    label = 11
                elif folderName in ['M']:
                    label = 12
                elif folderName in ['N']:
                    label = 13
                elif folderName in ['O']:
                    label = 14
                elif folderName in ['P']:
                    label = 15
                elif folderName in ['Q']:
                    label = 16
                elif folderName in ['R']:
                    label = 17
                elif folderName in ['S']:
                    label = 18
                elif folderName in ['T']:
                    label = 19
                elif folderName in ['U']:
                    label = 20
                elif folderName in ['V']:
                    label = 21
                elif folderName in ['W']:
                    label = 22
                elif folderName in ['X']:
                    label = 23
                elif folderName in ['Y']:
                    label = 24
                elif folderName in ['Z']:
                    continue
                elif folderName in ['del']:
                    continue
                elif folderName in ['nothing']:
                    continue
                elif folderName in ['space']:
                    continue
                else:
                    continue

                for i, image_filename in enumerate(os.listdir(path + '/' + folderName)):
                    img_file = cv2.imread(path + '/' + folderName + '/' + image_filename, 0)
                    if img_file is not None:
                        img_file = skimage.transform.resize(img_file, (28, 28))
                        img_arr = np.asarray(img_file)
                        samples.append(img_arr)
                        labels.append(mapping.index(label))
        # with open(path) as f:
        #     _ = next(f)
        #     for line in csv.reader(f):
        #         label = int(line[0])
        #         labels.append(mapping.index(label))
        #         samples.append(list(map(int, line[1:])))
        return labels, samples

    def __init__(self, path="data/asl_alphabet_train/asl_alphabet_train",mean=[0.485], std=[0.229]):
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

            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)])

        return {
            'image': transform(self._samples[idx]).float(),
            'label': torch.from_numpy(self._labels[idx]).float()
        }


def get_train_test_loader_asl(batch_size=32):

    trainset = SignLanguageASL('data/asl_alphabet_train/asl_alphabet_train')
    trainset, testset = train_test_split(trainset, test_size=0.2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # testset = SignLanguageASL('data/asl_alphabet_train/asl_alphabet_train', sign)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, trainset, testset


if __name__ == '__main__':
    loader, _, _, _ = get_train_test_loader_asl(2)
    print(next(iter(loader)))
