import torch 
import torch.nn as nn
'''
class WGAN_G(nn.Module):
    def __init__(self):
        super(WGAN_G, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block5 = nn.Sequential(
            nn.Linear(1024, 3*64*64),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(-1,3,64,64)
        return x

class WGAN_D(nn.Module):
    def __init__(self):
        super(WGAN_D, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Linear(3*64*64, 1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block4 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
'''
class WGAN_G(nn.Module):
    def __init__(self):
        super(WGAN_G, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

class WGAN_D(nn.Module):
    def __init__(self):
        super(WGAN_D, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x.mean(0).view(1)
        
def test():
    model = WGAN_G()
    model.eval()
    x = torch.rand(2,100).view(-1,100,1,1)
    output = model(x)
    print(output.shape)
    
    model2 = WGAN_D()
    model2.eval()
    output = model2(output)
    print(output.shape)

if __name__ == '__main__':
    test()
