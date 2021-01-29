"""
Create the dataset class.
Should also transform graphs to right size before storing
"""

from torch.utils.data.dataset import Dataset
from torchvision import transforms

class Graph_Dataset(Dataset):

    def __init__(self):

        self.transformations = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])
                                ])
        self.dataset = None  # csv file with names and labels

    def __getitem__(self, index):
        # index is the image number, ie the image name

        # ...
        return (img, label)

    def __len__(self):
        last_line = self.dataset.readlines()[-1]
        return last_line[0]