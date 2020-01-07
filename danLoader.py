import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image

class DIGIT(Dataset):
    def __init__(self, root_img, root_label, transform=None):
        self.images_filenames = glob.glob(os.path.join(root_img, '*.png'))
        self.transform = transform
        self.len = len(self.images_filenames)
        self.labels = parse_label(root_label)
        #windows
        self.images_filenames = sorted(self.images_filenames, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        
    def __getitem__(self, index):
        image_name = self.images_filenames[index]
        image = Image.open(image_name)
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, self.labels[index]
        
    def __len__(self):
        return self.len 
        
def parse_label(filename):
    result = []
    first_row = True
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            token = line.strip().split(',')
            if len(token) != 2 or first_row:
                first_row = False
                continue
            result.append(int(token[1]))
    return result  

def test_loader():
    trainset = DIGIT('hw3_data/digits/usps/train', 'hw3_data/digits/usps/train.csv', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
    img1, label = trainset[2]
    print(img1.shape, label)
    
if __name__ == '__main__':
    test_loader()
