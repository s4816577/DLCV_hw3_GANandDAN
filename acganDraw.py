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
import sys
import os

bt_size = 10
seed_ind = [19, 29, 105, 107, 177, 178, 182, 198, 241, 210]

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path,map_location='cuda')
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    
def test(model_G, model_D, device):
    model_G.eval()
    model_D.eval()
    '''
    fix_list = []
    for i in seed_ind:
        torch.manual_seed(i)
        fixedNoise = torch.randn((1, 100))
        fix_list.append(fixedNoise)
    fix_list = torch.stack(fix_list).to(device).view(-1, 100)
    
    fake_labels_0 = torch.randint(low=0, high=1, size=(bt_size, 1)).float().to(device)
    fake_labels_1 = torch.randint(low=1, high=2, size=(bt_size, 1)).float().to(device)
    z_0 = torch.cat((fix_list, fake_labels_0), dim=1).to(device)
    z_1 = torch.cat((fix_list, fake_labels_1), dim=1).to(device)
    z = torch.cat((z_0, z_1), dim=0).to(device)
    '''
    z = np.load('ac_latent.npy')
    z = torch.from_numpy(z).to(device)
    fake_imgs = model_G(z)
    save_image(fake_imgs.data[:20], sys.argv[1] + 'fig2_2.jpg', nrow=10, normalize=True)

    #save latent data
    #np.save('ac_latent', z.cpu().numpy())

def main():
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model_G = acganModels.ACGAN_G().to(device)
    model_D = acganModels.ACGAN_D().to(device)
    load_checkpoint('Models/ACGAN-G.pth', model_G, 111)
    load_checkpoint('Models/ACGAN-D.pth', model_D, 111)
    
    #Debug(model, testset_loader)
    test(model_G, model_D, device)

if __name__ == '__main__':
    main()
