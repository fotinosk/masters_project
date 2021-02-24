"""
Create the dataset class.
Should also transform graphs to right size before storing
"""

from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import pandas as pd
import PIL

class Graph_Dataset(Dataset):

    def __init__(self):

        self.transformations = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ])
        self.cwd = os.getcwd()
        self.image_files =  os.listdir(self.cwd + '/dataset/images')
        self.labels = pd.read_csv('./dataset/labels.csv')

    def __getitem__(self, index):

        img_name = self.labels.iloc[index-1,:][0]
        label = self.labels.iloc[index-1,:][1]
        image_name = os.path.join(self.cwd + '/dataset/images', img_name)
        image =  PIL.Image.open(image_name)
        image = self.transformations(image)
        return (image, label)

    def __len__(self):
        return int(self.labels.iloc[-1,:].name) + 2