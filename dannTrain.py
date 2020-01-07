import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dannModels
import dannLoader
import dannTest

bt_size = 64
    
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def train_mix(model_F, model_L, model_D, train_loader1, train_loader2, test_loader, device, epoch, need_exten_channel1, need_exten_channel2):
    optimizer = optim.Adam(list(model_F.parameters())+list(model_D.parameters())+list(model_L.parameters()), lr=0.0001)
    label_classifier_loss = nn.CrossEntropyLoss()
    domain_classifier_loss = nn.CrossEntropyLoss()
    max_acc = 0
    max_ep = 0
    for ep in range(epoch):
        start_steps = ep * len(train_loader1)
        total_steps = epoch * len(train_loader2)
        for batch_idx, (source_train_data, target_train_data) in enumerate(zip(train_loader1, train_loader2)):
            #train model_D
            #init
            source_imgs, source_labels = source_train_data
            target_imgs, target_labels = target_train_data
            source_labels = source_labels.to(device)
            target_labels = target_labels.to(device)
            source_imgs = source_imgs.to(device)
            target_imgs = target_imgs.to(device)
            if need_exten_channel1:
                source_imgs = torch.cat((source_imgs, source_imgs, source_imgs), 1).to(device)
            if need_exten_channel2:
                target_imgs = torch.cat((target_imgs, target_imgs, target_imgs), 1).to(device)
            combined_imgs = torch.cat((source_imgs, target_imgs), 0).to(device)
            source_domains = torch.zeros(source_labels.shape[0]).type(torch.LongTensor)
            target_domains = torch.ones(target_labels.shape[0]).type(torch.LongTensor)
            combined_domains = torch.cat((source_domains, target_domains), 0).to(device)
            
            model_F.train()
            model_L.train()
            model_D.train()
            optimizer.zero_grad()
            
            p = float(batch_idx + start_steps) / total_steps
            const = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            
            #get predict labels
            source_features = model_F(source_imgs)
            source_pre_labels = model_L(source_features)
            
            combined_features = model_F(combined_imgs)
            combined_pre_domains = model_D(combined_features, const)
            
            #loss
            label_loss = label_classifier_loss(source_pre_labels, source_labels)
            domain_loss = domain_classifier_loss(combined_pre_domains, combined_domains)
            total_loss = label_loss + domain_loss
            
            #postit
            total_loss.backward()
            optimizer.step()
            '''
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx*(len(source_imgs)+len(target_imgs)), len(train_loader1.dataset)+len(train_loader2.dataset),
                    100. * batch_idx*(len(source_imgs)+len(target_imgs)) / (len(train_loader1.dataset)+len(train_loader2.dataset)), total_loss.item() / bt_size))
            '''
        if ep % 1 == 0:
            print(ep)
            acc = dannTest.test(model_F, model_L, test_loader, device, False)
            save_checkpoint("models_DANN_mix2/F-%d.pth" % ep, model_F, optimizer)
            save_checkpoint("models_DANN_mix2/L-%d.pth" % ep, model_L, optimizer)
            save_checkpoint("models_DANN_mix2/D-%d.pth" % ep, model_D, optimizer)
            if acc > max_acc:
                max_acc = acc
                max_ep = ep
    print(max_ep, max_acc)

def train_source_only(model_F, model_L, model_D, train_loader, test_loader, device, epoch, need_exten_channel):
    optimizer = optim.Adam(list(model_F.parameters())+list(model_L.parameters()), lr=0.0001)
    classifier_loss  = nn.CrossEntropyLoss()
    max_acc = 0
    max_ep = 0
    for ep in range(epoch):
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            #train model_D
            #init
            labels = labels.to(device)
            imgs = imgs.to(device)
            if need_exten_channel:
                imgs = torch.cat((imgs, imgs, imgs), 1).to(device)
            
            model_F.train()
            model_L.train()
            optimizer.zero_grad()
            
            #get predict labels
            features = model_F(imgs)
            pre_labels = model_L(features)
            
            #loss
            label_loss = classifier_loss(pre_labels, labels)
            
            #postit
            label_loss.backward()
            optimizer.step()
            '''
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(imgs), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), label_loss.item() / bt_size))
            '''            
        if ep % 1 == 0:
            print(ep)
            acc = dannTest.test(model_F, model_L, test_loader, device, True)
            if acc > max_acc:
                max_acc = acc
                max_ep = ep
    print(max_ep, max_acc)
        
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
    
    #source only part
    #usps -> mnistm -> svhn
    #case1
    #source_train_set = dannLoader.DIGIT('hw3_data/digits/mnistm/train', 'hw3_data/digits/mnistm/train.csv', transforms.ToTensor())
    #target_test_set = dannLoader.DIGIT('hw3_data/digits/svhn/test', 'hw3_data/digits/svhn/test.csv', transforms.ToTensor())
    #train_loader = DataLoader(source_train_set, batch_size=bt_size, shuffle=True, num_workers=1)
    #test_loader = DataLoader(target_test_set, batch_size=bt_size, shuffle=False, num_workers=1)
    #case2
    source_train_set = dannLoader.DIGIT('hw3_data/digits/mnistm/train', 'hw3_data/digits/mnistm/train.csv', transforms.ToTensor())
    target_train_set = dannLoader.DIGIT('hw3_data/digits/svhn/train', 'hw3_data/digits/svhn/train.csv', transforms.ToTensor())
    target_test_set = dannLoader.DIGIT('hw3_data/digits/svhn/test', 'hw3_data/digits/svhn/test.csv', transforms.ToTensor())
    train_loader1 = DataLoader(source_train_set, batch_size=bt_size, shuffle=True, num_workers=1)
    train_loader2 = DataLoader(target_train_set, batch_size=bt_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(target_test_set, batch_size=bt_size, shuffle=False, num_workers=1)
    #case3
    #target_train_set = dannLoader.DIGIT('hw3_data/digits/usps/train', 'hw3_data/digits/usps/train.csv', transforms.ToTensor())
    #target_test_set = dannLoader.DIGIT('hw3_data/digits/usps/test', 'hw3_data/digits/usps/test.csv', transforms.ToTensor())
    #train_loader = DataLoader(target_train_set, batch_size=bt_size, shuffle=True, num_workers=1)
    #test_loader = DataLoader(target_test_set, batch_size=bt_size, shuffle=False, num_workers=1)
    
    #Debug(model, testset_loader)
    #train_source_only(model_F, model_L, model_D, train_loader, test_loader, device, 100, True)
    train_mix(model_F, model_L, model_D, train_loader1, train_loader2, test_loader, device, 100, False, False)

if __name__ == '__main__':
    main()
