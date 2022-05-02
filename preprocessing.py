from torch.utils.data import Dataset 
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn 
import numpy as np 
import torch 


import csv 

class SignLanguageMNIST(Dataset): 
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
    def extract(path:str): 
        """
        Extracts labels and samples from the CSV file
        28 x 28 = 784 pixel values per sample 

        """
        mapping = SignLanguageMNIST.map_to_alphabet()
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
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)])
        
        return {
            'image': transform(self._samples[idx]).float(),
            'label': torch.from_numpy(self._labels[idx]).float()
        }