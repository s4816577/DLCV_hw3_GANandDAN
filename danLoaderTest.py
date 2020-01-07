import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image

class DIGIT(Dataset):
    def __init__(self, root_img, transform=None):
        self.images_filenames = glob.glob(os.path.join(root_img, '*.png'))
        self.transform = transform
        self.len = len(self.images_filenames)
        
        #windows
        self.images_filenames = sorted(self.images_filenames, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        
    def __getitem__(self, index):
        image_name = self.images_filenames[index]
        image = Image.open(image_name)
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image
        
    def __len__(self):
        return self.len 

def test_loader():
    trainset = DIGIT('hw3_data/digits/usps/train', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]))
    img1, label = trainset[2]
    print(img1.shape, label)
    
if __name__ == '__main__':
    test_loader()
