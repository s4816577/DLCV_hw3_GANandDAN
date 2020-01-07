import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dannModels
import dannLoader

bt_size = 64

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path, map_location='cuda:1')
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

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
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda:1' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model_F = dannModels.Feature_Extractor().to(device)
    model_L = dannModels.Label_Classifier().to(device)
    model_D = dannModels.Domain_Classifier().to(device)
    load_checkpoint('mix/F-23.pth', model_F, 111)
    load_checkpoint('mix/L-23.pth', model_L, 111)
    load_checkpoint('mix/D-23.pth', model_D, 111)
    
    #usps -> mnistm -> svhn
    #dataset init and loader warper
    target_set = dannLoader.DIGIT('hw3_data/digits/mnistm/test', 'hw3_data/digits/mnistm/test.csv', transforms.ToTensor())
    target_loader = DataLoader(target_set, batch_size=bt_size, shuffle=True, num_workers=1)
    
    #Debug(model, testset_loader)
    test(model_F, model_L, target_loader, device, False)

if __name__ == '__main__':
    main()
