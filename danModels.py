import torch
import torch.nn as nn

class Feature_Extractor(nn.Module):
    def __init__(self):
        super(Feature_Extractor, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 10, 5),
            nn.BatchNorm2d(10),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Linear(10*4*4, 100),
            nn.BatchNorm1d(100),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(-1, 10*4*4)
        x = self.block3(x)
        return x

class Label_Classifier(nn.Module):
    def __init__(self):
        super(Label_Classifier, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        return x
        
def main():
    model = Feature_Extractor()
    print(model)
    
    model = Label_Classifier()
    print(model)
        
if __name__ == '__main__':
    main()