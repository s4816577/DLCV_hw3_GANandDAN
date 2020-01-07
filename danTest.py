import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import danModels
import danLoader

bt_size = 64

def test(model_F, model_L, test_loader, device, need_exten_channel):
    correct = 0
    with torch.no_grad():
        model_F.eval()
        model_L.eval()
        for batch_idx, (imgs, labels) in enumerate(test_loader):
            #train model_D
            #init
            labels = labels.to(device)
            imgs = imgs.to(device)
            if need_exten_channel:
                imgs = torch.cat((imgs, imgs, imgs), 1).to(device)
                
            #get predict labels
            features = model_F(imgs)
            pre_labels = model_L(features)
                
            #caculate correct
            pre_labels = pre_labels.data.max(1, keepdim=True)[1]
            correct += pre_labels.eq(labels.data.view_as(pre_labels)).cpu().sum()
        
                
    print('Accuracy: {}/{}\t\t\t({:.2f}%)'.format(
            correct, len(test_loader.dataset), 100. * correct.item() / len(test_loader.dataset)))
    return 100. * correct.item() / len(test_loader.dataset)

def main():
    '''
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda:1' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model_F = danModels.Feature_Extractor().to(device)
    model_L = danModels.Label_Classifier().to(device)
    model_D = danModels.Domain_Classifier().to(device)
    
    #dataset init and loader warper
    source_set = danLoader.DIGIT('hw3_data/digits/usps/train', 'hw3_data/digits/usps/train.csv', transforms.ToTensor())
    target_set = danLoader.DIGIT('hw3_data/digits/mnistm/train', 'hw3_data/digits/mnistm/train.csv', transforms.ToTensor())
    source_loader = DataLoader(source_set, batch_size=bt_size, shuffle=True, num_workers=1)
    target_loader = DataLoader(target_set, batch_size=bt_size, shuffle=True, num_workers=1)
    
    #Debug(model, testset_loader)
    train_source(model_F, model_L, model_D, source_loader, target_loader, device, 10000, True)
    '''

if __name__ == '__main__':
    main()
