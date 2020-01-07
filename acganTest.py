#pick 595
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch import autograd
import numpy as np
import acganModels
import acganLoader

bt_size = 10

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path,map_location='cpu')
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def test(model_G, model_D, real_img_loader, device):
    #load models
    load_checkpoint('models_ACWGANDC/G-3340.pth', model_G, 111)
    load_checkpoint('models_ACWGANDC/D-3340.pth', model_D, 111)
    '''
    for batch_idx, (real_imgs, real_labels) in enumerate(real_img_loader):
        if batch_idx > 1:
            break
        valid, cls = model_D(real_imgs)
        print(valid)
        print('gt', real_labels)
        print('pre', cls)
    '''
    #gen 0 z
    fake_labels = torch.randint(low=0, high=1, size=(bt_size, 1)).float().to(device)
    latent_data = torch.zeros((bt_size, 100)).normal_(0,1).normal_(0,1).to(device)
    z_0 = torch.cat((latent_data, fake_labels), dim=1).to(device)
    
    #gen 1 z
    fake_labels = torch.randint(low=1, high=2, size=(bt_size, 1)).float().to(device)
    z_1 = torch.cat((latent_data, fake_labels), dim=1).to(device)
    
    #cat z_0 z_1
    z = torch.cat((z_0, z_1), dim=0).to(device)
    
    fake_imgs = model_G(z)
    valid, cls = model_D(fake_imgs)
    save_image(fake_imgs.data[:20], "images_Test/0.png", nrow=10, normalize=True)

    #save latent data
    np.save('latent', latent_data.cpu().numpy())

def main():
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cpu' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model_G = acganModels.ACGAN_G().to(device)
    model_D = acganModels.ACGAN_D().to(device)
    
    #real_data
    real_img_set = acganLoader.FACE('hw3_data/face/train', 'hw3_data/face/train.csv', transforms.ToTensor())
    real_img_loader = DataLoader(real_img_set, batch_size=bt_size, shuffle=False, num_workers=1)
    
    #Debug(model, testset_loader)
    test(model_G, model_D, real_img_loader, device)

if __name__ == '__main__':
    main()
