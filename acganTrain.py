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

bt_size = 64
n_critic = 5
    
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def train(model_G, model_D, real_img_loader, device, epoch):
    #optimizer_G = torch.optim.Adam(model_G.parameters(), lr=0.00005, eps=0.001, weight_decay=0.0001)
    #optimizer_D = torch.optim.SGD(model_D.parameters(), lr=0.00005, momentum=0.9, weight_decay=0.0001)
    optimizer_G = torch.optim.RMSprop(model_G.parameters(), lr=0.00005)
    optimizer_D = torch.optim.RMSprop(model_D.parameters(), lr=0.00005)
    labels_mse_loss = nn.BCELoss()
    #optimizer_G = torch.optim.Adam(model_G.parameters(), lr=0.0001, betas=(0, 0.9))
    #optimizer_D = torch.optim.Adam(model_D.parameters(), lr=0.0001, betas=(0, 0.9))
    
    for ep in range(epoch):
        for batch_idx, (real_imgs, real_labels) in enumerate(real_img_loader):
            #train model_D
            #init
            for p in model_D.parameters(): 
                p.requires_grad = True
            for p in model_D.parameters():
                p.data.clamp_(-0.01, 0.01)
            optimizer_D.zero_grad()
            
            #get real_img, fake_img, real_labels, fake_labels
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.view(bt_size,-1).to(device)
            fake_labels = torch.randint(low=0, high=2, size=(bt_size, 1)).float().to(device)
            latent_data = torch.zeros((bt_size, 100)).normal_(0,1).to(device)
            z = torch.cat((latent_data, fake_labels), dim=1).to(device)
            fake_imgs = model_G(z)
            
            #loss
            real_valid, real_cls = model_D(real_imgs)
            fake_valid, fake_cls = model_D(fake_imgs)
            D_loss = (-torch.mean(real_valid) + torch.mean(fake_valid) + labels_mse_loss(real_cls, real_labels) + labels_mse_loss(fake_cls, fake_labels)) / 4.0
            
            #postit
            D_loss.backward()
            optimizer_D.step()
            
            if batch_idx % n_critic == 0:
                #train model_G
                #init
                for p in model_D.parameters():
                    p.requires_grad = False
                optimizer_G.zero_grad()
                #get predict valid and cls
                valid, cls = model_D(model_G(z))
                #loss
                G_loss = (-torch.mean(valid) + labels_mse_loss(cls, fake_labels)) / 2.0
                #postit
                G_loss.backward()
                optimizer_G.step()
                
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f}\tD_Loss: {:.6f}'.format(
                    ep, batch_idx * len(real_imgs), len(real_img_loader.dataset),
                    100. * batch_idx / len(real_img_loader), G_loss.item() / bt_size, D_loss.item() / bt_size))
                    
        if ep % 5 == 0:
            fake_imgs = model_G(z)
            save_image(fake_imgs.data[:36], "images_ACWGANDC/%d.png" % ep, nrow=6, normalize=True)
            save_checkpoint("models_ACWGANDC/G-%d.pth" % ep, model_G, optimizer_G)
            save_checkpoint("models_ACWGANDC/D-%d.pth" % ep, model_D, optimizer_D)
        
def main():
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda:1' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model_G = acganModels.ACGAN_G().to(device)
    model_D = acganModels.ACGAN_D().to(device)
    
    #dataset init and loader warper
    real_img_set = acganLoader.FACE('hw3_data/face/train', 'hw3_data/face/train.csv', transforms.ToTensor())
    real_img_loader = DataLoader(real_img_set, batch_size=bt_size, shuffle=True, num_workers=1)
    
    #Debug(model, testset_loader)
    train(model_G, model_D, real_img_loader, device, 10000)

if __name__ == '__main__':
    main()
