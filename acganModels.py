import torch 
import torch.nn as nn

class ACGAN_G(nn.Module):
    def __init__(self):
        super(ACGAN_G, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(101, 512*4*4),
            nn.BatchNorm1d(512*4*4),
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
        x = x.view(-1, 512, 4, 4)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

class ACGAN_D(nn.Module):
    def __init__(self):
        super(ACGAN_D, self).__init__()
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
        self.block_realfake = nn.Sequential(
            nn.Linear(512*4*4,1)
        )
        self.block_cls = nn.Sequential(
            nn.Linear(512*4*4,1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        realfake = self.block_realfake(x)
        cls = self.block_cls(x)
        return realfake, cls
        
def test():
    model = ACGAN_G()
    model.eval()
    x = torch.rand(3,101)
    output = model(x)
    print(output.shape)
    
    model2 = ACGAN_D()
    model2.eval()
    realfake, cls = model2(output)
    #print(model2)

if __name__ == '__main__':
    test()