#!/usr/bin/env python
import os
import glob
import torch
import torchvision
import pandas as pd
import numpy as np
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.multiprocessing as mp
import torch.optim as optim
from osgeo import gdal
from torch.utils.data.sampler import SubsetRandomSampler
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Read in csv for training data
path = '/scratch/07043/ac1824/EE_Super'
path_train = os.path.join(path,"ls5_df_train_meanCorrect.csv")
path_train = glob.glob(path_train)[0]

# Read in csv for testing data (not used here)
path_test = os.path.join(path,"ls5_df_test_meanCorrect.csv")
path_test = glob.glob(path_test)[0]
df = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

class rasterDataset(Dataset):
    
    def __init__(self,csv,transform=None):
        self.raster_csv = pd.read_csv(csv)
        self.transform = transform
    def __len__(self):
        return len(self.raster_csv)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Column for 10m images
        low_path = self.raster_csv.iloc[idx]['path10m_real']
        # Column for 1m images
        high_path = self.raster_csv.iloc[idx]['path1m_real']
        # Read images
        temp_img = gdal.Open(low_path)
        img_low = np.array(temp_img.GetRasterBand(1).ReadAsArray())
        temp_h_img = gdal.Open(high_path)
        img_high = np.array(temp_h_img.GetRasterBand(1).ReadAsArray())
        sample = {'low_res':img_low,'high_res':img_high,
                 'path1m':high_path,'path10m':low_path}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToTensor(object):
    
    def __call__(self,sample):
        img_low, img_high = sample['low_res'],sample['high_res']
        # Add another dimension to the low and high resolution images
        img_low = img_low[:,:,None]
        img_high = img_high[:,:,None]
        # Move the channel dimension to the first dimension in the array
        img_low = img_low.transpose([2,0,1])
        img_high = img_high.transpose([2,0,1])
        # Check the mean of each image. Used to eliminate any images with NAN if not
        # already filtered out (which they should be)
        img_l_mean = np.mean(img_low)
        img_h_mean = np.mean(img_high)
        if not np.isnan(img_l_mean) and not np.isnan(img_h_mean):
            if not np.isinf(img_l_mean) and not np.isinf(img_h_mean):
                img_l_std = np.std(img_low)
                # Transform the numpy arrays into pytorch tensors and place on the GPU
                img_low = torch.tensor(img_low,device=torch.device('cuda:0')).type(torch.cuda.FloatTensor)
                img_high = torch.tensor(img_high,device=torch.device('cuda:0')).type(torch.cuda.FloatTensor)
                return {'img_low':img_low,
                        'img_high':img_high}

class Conv1stLayer(nn.Module):
    def __init__(self, in_c, out_c,kernel,
                 stride_in,padding_in,act_str='relu'):
        super(Conv1stLayer, self).__init__()
        # 2D convolution
        self.conv1 = nn.Conv2d(in_c,out_c,kernel_size=kernel,
                              stride=stride_in,padding=padding_in)
        self.act_str = act_str
        self.activation = nn.ReLU()
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.activation(out)
        return out

class DeConv(nn.Module):
    def __init__(self, in_c, out_c,kernel,
                 stride_in,padding_in,out_shape):
        super(DeConv,self).__init__()
        # Transposed convolution for final two layers
        self.deconv1 = nn.ConvTranspose2d(in_c,out_c,
                                          kernel_size=kernel,
                                          stride=stride_in,padding=padding_in)
        # Specifying an output shape to ensure pytorch keeps the dimensions consistent with 
        # the high resolution, ground truth image.
        self.out_shape = out_shape
    def forward(self,x):
        out = self.deconv1(x,self.out_shape)
        return out

class ResBlock(nn.Module):
    def __init__(self,in_c,out_c,kernel,stride_in,padding_in,act_str):
        super(ResBlock,self).__init__()
        # Start ResBlock with a shape preserving convolution
        self.conv1 = nn.Conv2d(in_c,out_c,kernel_size=kernel,
                              stride=stride_in,padding=padding_in)
        if act_str == 'elu':
            self.activation = nn.ELU()
            
        if act_str == 'relu':
            self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(in_c,out_c,kernel_size=kernel,
                              stride=stride_in,padding=padding_in)
        
    def forward(self,x):
        # Convolve + activation
        img = self.conv1(x)
        img = self.activation(img)
        # Convolve the result
        img = self.conv2(img)
        # Add the input feature maps to the output
        residual = img + x
        return residual

class Attention(nn.Module):
    def __init__(self,in_dim,activation):
        super(Attention,self).__init__()
        self.channel_in=in_dim
        self.activation = activation
        # Define querries through a feature map reducing convolution, shape preserved
        self.query_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=3,padding=1)
        # Define value convolution
        self.key_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=3,padding=1)
        # Since values are multiplied to the softmax of (QK^T), keep input dimensions equal
        # to what was input
        self.value_conv=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=3,padding=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        batch_size,channels,width,height = x.size()
        # Flatten the input feature maps after the convolution for the queries transposed
        proj_query  = self.query_conv(x).view(batch_size,-1,width*height).permute(0,2,1)
        # Flatten the input feature maps after the convolution for the values
        proj_key =  self.key_conv(x).view(batch_size,-1,width*height)
        # Next two lines compute how much each pixel across all feature maps influence any single pixel
        energy =  torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
        # Value convolution
        proj_value = self.value_conv(x).view(batch_size,-1,width*height)
        # Find pixels with highest probability of having a dependency on any other pixel
        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(batch_size,channels,width,height)
        # Add input to attention head to the output as a residual connection
        out = self.gamma*out + x
        return out

class AlecNet(nn.Module):
    def __init__(self,n_channels,features,channel_scale_in,
                 out_shape,out_shape2,out_shape3,act,sequence_scale=True):
        super(AlecNet,self).__init__()
        self.features=features
        self.channel_scale = channel_scale_in
        # (Batch_size x n_channels x 75 x 75)
        self.conv1 = Conv1stLayer(in_c=1,out_c=n_channels,
                              kernel=3,padding_in=1,stride_in=1,
                              act_str=act)
        # (Batch_size x features x 75 x 75)
        self.conv2 = Conv1stLayer(n_channels,features,kernel=3,
                              padding_in=1,stride_in=1,
                              act_str=act)
        # (Batch_size x features x 75 x 75)
        self.res = ResBlock(features,features,kernel=3,
                            stride_in=1,padding_in=1,act_str=act)
        # (Batch_size x 1 x 75 x 75): Used to aggregate feature maps at the end of the model
        self.group = Conv1stLayer(in_c=features//4,out_c=1,kernel=1,
                              padding_in=0,stride_in=1,
                              act_str=None)
        # (Batch_size x features//4 x 75 x 75): Reduce features to match output number of features
        # from the attention head.
        self.groupRes = Conv1stLayer(in_c=features,out_c=features//4,kernel=1,
                              padding_in=0,stride_in=1,
                              act_str='relu')
        # (Batch_size x features//4 x 15 x 15): Reduce features and size from last standard
        # residual cell for computational complexity purposes.
        self.preattention = Conv1stLayer(features,features//4,kernel=3,stride_in=5,
                              padding_in=0,act_str=act)
        # Attention head
        self.attention = Attention(features//4,'relu')
        # (Batch_size x features//4 x 75 x 75): Reduce features from last standard residual cell
        self.postattention = DeConv(features//4,features//4,kernel=5,stride_in=5,
                              padding_in=0,out_shape=out_shape)
        # (Batch_size x 1 x 375 x 375): Reduce features from last standard residual cell
        self.upscale1 = DeConv(1,1,kernel=5,stride_in=5,
                              padding_in=0,out_shape=out_shape2)
        # (Batch_size x 1 x 750 x 750): Reduce features from last standard residual cell
        self.upscale2 = DeConv(1,1,kernel=2,stride_in=2,
                               padding_in=0,out_shape=out_shape3)
        
    def forward(self,x):
        # Convolution Block
        img = self.conv1(x)
        img = self.conv2(img)
        # Residual Block
        res1 = self.res(img)
        res2 = self.res(res1)
        res3 = self.res(res2)
        res4 = self.res(res3)
        res5 = self.res(res4)
        res6 = self.groupRes(res5)
        # Self-Attention Block
        pre_att = self.preattention(res5)
        att = self.attention(pre_att)
        post_att = self.postattention(att)
        # Upscale Block
        group1 = self.group(post_att+res6)
        up1 = self.upscale1(group1)
        up2 = self.upscale2(up1)
        return up2
        
if __name__ == "__main__":
    # Read custom pytorch dataset class
    data = rasterDataset(path_train,transform=transforms.Compose([ToTensor()]))
    # Open up two test images to get important shape features for the input.
    data_t = data[0]['img_low'][None,:]
    high_img = data[0]['img_high'][None,:]
    # Low resolution image size (75,75)
    low_shape=data_t.size()
    # 5x the low resolution size: 2nd to last upsampling layer
    out_shape_half = high_img[:,:,:int(high_img.shape[2]/2),
                                    :int(high_img.shape[3]/2)].size()
    # Final Image size
    out_shape_final = data[0]['img_high'][None,:].size()
    # Training parameters
    valid_split = 0.2
    random_seed=42
    batch_size=24
    dataset_size=len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_split*dataset_size))
    shuffle_dataset=True
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    # Create sampler for training and validation data, so the validation set can be evaluated 
    # at the same time as the training.
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler=SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    # Create separate dataloaders for training and validation with random sampling
    alecloader = DataLoader(data,batch_size=batch_size,sampler=train_sampler)
    validloader = DataLoader(data,batch_size=batch_size,sampler=valid_sampler)
    # Instantiate Model class
    model = AlecNet(n_channels=64,features=32,channel_scale_in=16,
                   out_shape=low_shape,out_shape2=out_shape_half,out_shape3=out_shape_final,
                   act='relu')
    # Used only on the first epoch, but it finds optimal configuration/compute routines
    # for the convolution operations in the model. Typically, performance speeds up by 
    # by a factor of approximately 4-5x for all epochs after the first
    torch.backends.cudnn.benchmark=True
    model.cuda()
    model.zero_grad()
    # Warm up the GPU
    for c in range(1):
        test_c1 = nn.Conv2d(1,64,kernel_size=3,padding=1,stride=1)
        test_c2 = nn.Conv2d(64,32,kernel_size=3,padding=1,stride=1)
        test_c3 = nn.Conv2d(32,32,kernel_size=3,padding=1,stride=1)
        c1 = test_c1(torch.randn(batch_size,1,75,75))
        c2 = test_c2(c1)
        c3 = test_c3(c2)
        test_c3 = nn.ConvTranspose2d(1,1,kernel_size=10,stride=10,padding=0)
        d = test_c3(torch.randn(batch_size,1,75,75))
    # Output path
    model_path = '/scratch/07043/ac1824/EE_Super/resnet_models_v2/results_150k'
    # Optimizer and learning rate
    optimizer = torch.optim.Adam(params=model.parameters(),lr=.001)
    for param_g in optimizer.param_groups:
        start_lr = param_g['lr']
    criterion = nn.L1Loss()
    #criterion.cuda()
    torch.cuda.synchronize()
    for i in range(100):
        epoch_loss = 0
        valid_loss = 0
        t1 = time.perf_counter()
        print(f'Epoch: {i}')
        for num,sample in enumerate(alecloader):
            torch.cuda.synchronize()
            t_train = time.perf_counter()
            low_img, high_img = sample['img_low'],sample['img_high']
            # Model
            pred = model(low_img)
            # Loss
            loss = criterion(pred,high_img)
            # Used to get a per epoch loss by diviving by the resulting sum, by the number of batch
            # in the epoch, i.e. average epoch loss.
            epoch_loss+=loss.item()
            loss.backward()
            optimizer.step()
            for param in model.parameters():
                param.grad=None
            torch.cuda.synchronize()
            t_train2 = time.perf_counter()
        train_e_loss = epoch_loss/(num+1)
        print(train_e_loss)
        t2 = time.perf_counter()

        # Validation set evaluation at the end of each epoch
        model.eval()
        # prevent back propagation on validation set
        with torch.no_grad():
            for num2,sample2 in enumerate(validloader):
                t_v1 = time.perf_counter()
                low_img2, high_img2 = sample2['img_low'],sample2['img_high']
                t_v2 = time.perf_counter()
                pred_v = model(low_img2)
                loss_v = criterion(pred_v,high_img2)
                valid_loss+=loss_v.item()
        valid_e_loss = valid_loss/(num2+1)
        if valid_e_loss<=15 and optimizer.param_groups[0]['lr']==start_lr:
             for a in optimizer.param_groups:
                 a['lr'] = 0.1*a['lr']
        # Save the model at the end of every epoch. 
        model_name_pt = 'model_state_epoch_'+str(i)+'.pt'
        model_name_pt = os.path.join(model_path,model_name_pt)
        torch.save({'epoch':i,'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'model_loss':loss,'loss_train':train_e_loss,
                    'loss_valid':valid_e_loss},model_name_pt)
