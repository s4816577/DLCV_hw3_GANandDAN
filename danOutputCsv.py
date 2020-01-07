import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import danModels
import danLoaderTest
import sys
import os
import csv
import glob

images_filenames = glob.glob(os.path.join(sys.argv[1], '*.png'))
images_filenames = sorted(images_filenames, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
bt_size = 1

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def test(model_F, model_L, test_loader, device, need_exten_channel):
    with open(sys.argv[3], 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'label'])
        with torch.no_grad():
            model_F.eval()
            model_L.eval()
            for batch_idx, (imgs) in enumerate(test_loader):
                #train model_D
                #init
                imgs = imgs.to(device)
                if need_exten_channel:
                    imgs = torch.cat((imgs, imgs, imgs), 1).to(device)
                    
                #get predict labels
                features = model_F(imgs)
                pre_labels = model_L(features)
                pre_labels = pre_labels.data.max(1, keepdim=True)[1]
                
                #make current filename
                current_output_filename = images_filenames[batch_idx]
                current_output_filename = current_output_filename.split('/')[-1]
                
                #write to csv
                writer.writerow([current_output_filename, int(pre_labels.view(-1).cpu().data.numpy())])
        
def main():
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model_F = danModels.Feature_Extractor().to(device)
    model_L_src = danModels.Label_Classifier().to(device)
    model_L_tgt = danModels.Label_Classifier().to(device)
    need_exten_channel = False
    if sys.argv[2] == 'mnistm':
        load_checkpoint('Models/DAN1-F.pth', model_F, 111)
        load_checkpoint('Models/DAN1-L1.pth', model_L_src, 111)
        load_checkpoint('Models/DAN1-L2.pth', model_L_tgt, 111)
        need_exten_channel = False
    elif sys.argv[2] == 'svhn':
        load_checkpoint('Models/DAN2-F.pth', model_F, 111)
        load_checkpoint('Models/DAN2-L1.pth', model_L_src, 111)
        load_checkpoint('Models/DAN2-L2.pth', model_L_tgt, 111)
        need_exten_channel = False
    elif sys.argv[2] == 'usps':
        load_checkpoint('Models/DAN3-F.pth', model_F, 111)
        load_checkpoint('Models/DAN3-L1.pth', model_L_src, 111)
        load_checkpoint('Models/DAN3-L2.pth', model_L_tgt, 111)
        need_exten_channel = True
    
    #usps -> mnistm -> svhn
    #dataset init and loader warper
    target_set = danLoaderTest.DIGIT(sys.argv[1], transforms.ToTensor())
    target_loader = DataLoader(target_set, batch_size=bt_size, shuffle=False, num_workers=1)
    
    #Debug(model, testset_loader)
    test(model_F, model_L_src, target_loader, device, need_exten_channel)

if __name__ == '__main__':
    main()
