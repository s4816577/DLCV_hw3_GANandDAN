import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import danModels
import danLoader

bt_size = 1
data_num = 2000

COLOR_POOL = ('black', 'orange', 'purple', 'red', 'sienna', 'green', 'blue', 'grey', 'pink', 'yellow')

def plot_embedding(X, y, d):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig1 = plt.figure(figsize=(10, 10))
    for i in range(len(d)): 
        colors = COLOR_POOL[y[i]]
        plt.scatter(X[i, 0], X[i, 1], color=colors)
    fig_name = '4_case1_1-tsne.png'
    plt.savefig(fig_name)
    print('{} is saved'.format(fig_name))       
    
    fig2 = plt.figure(figsize=(10, 10))
    for i in range(len(d)): 
        if d[i] == 0:
            colors = (0.0, 0.0, 1.0, 1.0)
        else:
            colors = (1.0, 0.0, 0.0, 1.0)
        plt.scatter(X[i, 0], X[i, 1], color=colors)
    fig_name = '4_case1_2-tsne.png'
    plt.savefig(fig_name)
    print('{} is saved'.format(fig_name))

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path, map_location='cuda:1')
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def draw(model_F, model_L_src, model_L_tgt, source_test_loader, target_test_loader, device, need_exten_channel1, need_exten_channel2):
    source_imgs_list = []
    target_imgs_list = []
    source_label_list = []
    target_label_list = []
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    model_F.eval()
    model_L_src.eval()
    model_L_tgt.eval()
    with torch.no_grad():
        for batch_idx, (source_test_data, target_test_data) in enumerate(zip(source_test_loader, target_test_loader)):
            if batch_idx >= data_num:
                break
            source_imgs, source_labels = source_test_data
            target_imgs, target_labels = target_test_data
            source_imgs = source_imgs.to(device)
            target_imgs = target_imgs.to(device)
            source_labels = source_labels.to(device)
            target_labels = target_labels.to(device)
            
            if need_exten_channel1:
                source_imgs = torch.cat((source_imgs, source_imgs, source_imgs), 1).to(device)
            if need_exten_channel2:
                target_imgs = torch.cat((target_imgs, target_imgs, target_imgs), 1).to(device)
            source_imgs_list.append(source_imgs[0])
            target_imgs_list.append(target_imgs[0])
            source_label_list.append(source_labels[0])
            target_label_list.append(target_labels[0])
            
    source_imgs_list = torch.stack(source_imgs_list)
    target_imgs_list = torch.stack(target_imgs_list)
    combined_imgs_list = torch.cat((source_imgs_list, target_imgs_list), 0)
    combined_features_list = model_F(combined_imgs_list)
    combined_features_tsne = tsne.fit_transform(combined_features_list.detach().cpu().numpy())
    
    source_domain_list = torch.zeros(data_num).type(torch.LongTensor)
    target_domain_list = torch.ones(data_num).type(torch.LongTensor)
    combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0)
    
    source_label_list = torch.stack(source_label_list)
    target_label_list = torch.stack(target_label_list)
    combined_label_list = torch.cat((source_label_list, target_label_list), 0)
    plot_embedding(combined_features_tsne, combined_label_list.detach().cpu().numpy(), combined_domain_list)
    print(combined_features_tsne.shape, combined_domain_list.shape, combined_label_list.shape)
        
def main():
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda:1' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #usps -> mnistm -> svhn
    #model create and to device
    model_F = danModels.Feature_Extractor().to(device)
    model_L_src = danModels.Label_Classifier().to(device)
    model_L_tgt = danModels.Label_Classifier().to(device)
    load_checkpoint('mix/dan/F-35.pth', model_F, 111)
    load_checkpoint('mix/dan/L1-35.pth', model_L_src, 111)
    load_checkpoint('mix/dan/L2-35.pth', model_L_tgt, 111)
    
    #dataset init and loader warper
    source_test_set = danLoader.DIGIT('hw3_data/digits/usps/test', 'hw3_data/digits/usps/test.csv', transforms.ToTensor())
    test_loader1 = DataLoader(source_test_set, batch_size=bt_size, shuffle=False, num_workers=1)
    target_test_set = danLoader.DIGIT('hw3_data/digits/mnistm/test', 'hw3_data/digits/mnistm/test.csv', transforms.ToTensor())
    test_loader2 = DataLoader(target_test_set, batch_size=bt_size, shuffle=False, num_workers=1)
    
    #Debug(model, testset_loader)
    draw(model_F, model_L_src, model_L_tgt, test_loader1, test_loader2, device, True, False)

if __name__ == '__main__':
    main()
