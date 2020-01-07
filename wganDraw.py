import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch import autograd
import numpy as np
import wganModels
import wganLoader
import sys
import os

seed_ind = [5, 6, 19, 26, 38, 41, 45, 56, 48, 49, 52, 59, 64, 65, 71, 75, 99, 104, 131, 132, 133, 134, 140, 141, 149, 150, 154, 155, 157, 173, 180, 198]

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path,map_location='cuda')
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    
def test(model_G, model_D, device):
    model_G.eval()
    model_D.eval()
    '''
    z = []
    for i in seed_ind:
        torch.manual_seed(i-1)
        fixedNoise = torch.randn((1, 100))
        z.append(fixedNoise)
    z = torch.stack(z).to(device).view(-1,100,1,1)
    '''
    z = np.load('w_latent.npy')
    z = torch.from_numpy(z).to(device)
    fake_imgs = model_G(z)
    save_image(fake_imgs.data[:32], sys.argv[1] + 'fig1_2.jpg', nrow=8, normalize=True)

    #save latent data
    #np.save('w_latent', z.cpu().numpy())

def main():
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model_G = wganModels.WGAN_G().to(device)
    model_D = wganModels.WGAN_D().to(device)
    load_checkpoint('Models/WGAN-G.pth', model_G, 111)
    load_checkpoint('Models/WGAN-D.pth', model_D, 111)
    
    if not os.path.exists(sys.argv[1]):
        os.mkdir(sys.argv[1])
    
    #Debug(model, testset_loader)
    test(model_G, model_D, device)

if __name__ == '__main__':
    main()
