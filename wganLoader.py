import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
from PIL import Image

class FACE(Dataset):
    def __init__(self, root, transform=None):
        self.images_filenames = glob.glob(os.path.join(root, '*.png'))
        self.transform = transform
        self.len = len(self.images_filenames)
        
        #windows
        self.images_filenames = sorted(self.images_filenames, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        
    def __getitem__(self, index):
        image_name = self.images_filenames[index]
        
        image = Image.open(image_name)
        #image = image.resize((32,32), Image.ANTIALIAS)
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image
        
    def __len__(self):
        return self.len 

def test_loader():
    trainset = FACE('hw3_data/face/train')
    img1 = trainset[0]
    img1.save('haha.png')
    
if __name__ == '__main__':
    test_loader()
