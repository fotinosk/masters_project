import torch
import torchvision.models as models
from torchvision import transforms, utils, datasets
from data_generator import Generator
import torch.nn as nn
from PIL import Image
import matplotlib
import numpy as np

device = torch.device("cuda")

def generate_image_batches():
    list_labels = None
    list_images = None
    return list_labels, list_images

label, graph = Generator.generate_graph()

def prep_image(graph):
    array = graph.numpy()
    image = Image.fromarray(array)
    image = image.resize((224, 224))
    return image 

