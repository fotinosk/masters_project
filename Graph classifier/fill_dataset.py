import csv
import torch 
from data_generator import Generator
import os
from tqdm import tqdm
from PIL import Image

os.chdir(os.getcwd() + '/dataset')

def save_image(graph, name):
    array = graph.numpy()
    image = Image.fromarray(array)
    image = image.convert('L')
    image = image.resize((224, 224))
    image.save(name)
    # return image 

with open('labels.csv', 'a', newline='\n') as f:
    writer = csv.writer(f)

    for i in tqdm(range(5000)):
        label, graph = Generator.generate_graph()
        name = str(i) + '.png'
        save_image(graph, name)
        writer.writerow([name, label])