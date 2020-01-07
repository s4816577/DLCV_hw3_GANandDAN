import torch
import torch.nn as nn

class Reverse_Grade(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.const
        return output, None

class Feature_Extractor(nn.Module):
    def __init__(self):
        super(Feature_Extractor, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 70, 5, 1, 2),
            nn.BatchNorm2d(70),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(70, 50, 5, 1, 2),
            nn.BatchNorm2d(50),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(-1, 50*7*7)
        return x

class Label_Classifier(nn.Module):
    def __init__(self):
        super(Label_Classifier, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Linear(50*7*7, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class Domain_Classifier(nn.Module):
    def __init__(self):
        super(Domain_Classifier, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Linear(50*7*7, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x, const):
        x = Reverse_Grade.apply(x, const)
        x = self.block1(x)
        x = self.block2(x)
        return x

def main():
    model = Feature_Extractor()
    model.eval()
    x = torch.rand(3,3*28*28).view(-1,3,28,28)
    output = model(x)
    print(output.shape)
    model = Label_Classifier()
    print(model)
    model = Domain_Classifier()
    model.eval()
    output = model(output, 10)
    print(model)
        
if __name__ == '__main__':
    main()