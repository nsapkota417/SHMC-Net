import torch
import torch.nn as nn
import torchvision
from skimage.io import imread, imsave, imshow
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.utils as vutils
import numpy as np
import torch
import os
import subprocess
import wandb
import os
import yaml
import collections
from datetime import datetime
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import Dataset, DataLoader, random_split, Subset

class block(nn.Module):
    def __init__(self, 
                  in_channels, 
                  out_channels,
                  stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # all conv before BN no bias
        self.conv2d_1x1_1 = torch.nn.Conv2d(in_channels=in_channels, 
                                            out_channels=out_channels//4, 
                                            kernel_size=1, 
                                            stride=1, 
                                            padding=0,
                                            bias=False)

        self.BN1 = torch.nn.BatchNorm2d(num_features=out_channels//4)

        self.relu1 = torch.nn.ReLU(inplace=True)

        self.conv2d_3x3 = torch.nn.Conv2d(in_channels=out_channels//4, 
                                         out_channels=out_channels//4, 
                                         kernel_size=3, 
                                         stride=stride, 
                                         padding=1,
                                         bias=False)

        self.BN2 = torch.nn.BatchNorm2d(num_features= out_channels//4)

        self.relu2 = torch.nn.ReLU(inplace=True)

        self.conv2d_1x1_2 = torch.nn.Conv2d(in_channels=out_channels//4, 
                                         out_channels=out_channels, 
                                         kernel_size = 1, 
                                         stride=1, 
                                         padding=0,
                                         bias=False)
        
        self.BN3 = torch.nn.BatchNorm2d(num_features=out_channels)

        self.relu3 = torch.nn.ReLU(inplace=True)
        
        self.res_layer = nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0) if self.stride == 2 else nn.Identity(),
            torch.nn.Conv2d(in_channels=in_channels, 
                            out_channels=out_channels, 
                            kernel_size=1, 
                            stride=1, 
                            padding=0) if in_channels != out_channels else nn.Identity()
        )
        
       
    def forward(self, x):
        # print(f'block start: {x.shape}')
        x1 = x
        # print(f'x shape after: {x.shape}')
        # print(f'x1 shape before: {x1.shape}')
        x = self.conv2d_1x1_1(x)
        x = self.BN1(x)
        x = self.relu1(x)
        x = self.conv2d_3x3(x)
        x = self.BN2(x)
        x = self.relu2(x)
        x = self.conv2d_1x1_2(x)
        x = self.BN3(x)
        
        # print(f'x1 shape after: {x1.shape}')
        # import IPython; IPython.embed()
        x1 = self.res_layer(x1)
        # x1 = self.res_ap(x1)
        
        # x1 = self.res_conv(x1)
        # print(f'x1 shape after: {x1.shape}')
        x = x + x1
        x = self.relu3(x)
        # print(f'block end: {x.shape}')
        return x

class stage(nn.Module):
    def __init__(self, 
                  in_channels, 
                  out_channels,
                  num_stage=3,
                  stride =1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stage = num_stage
        self.stride = stride
        blocks = []
        blocks.append(block(in_channels=self.in_channels, 
                            out_channels=self.out_channels, stride=stride))
        for idx in range(self.num_stage-1):
            blocks.append(block(in_channels=self.out_channels, out_channels=self.out_channels, stride=1))
        self.blocks = nn.ModuleList(blocks)
        
    # def _getBlocks(self):
    #     return self.blocks
    
    def forward(self,x):
        # print(f'stage start: {x.shape}')
        for idx in range(len(self.blocks)):
            x = self.blocks[idx](x)
        # print(f'stage end: {x.shape}\n')
        return x

class ResNet(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, 
                            out_channels=64, 
                            kernel_size=7, 
                            stride=2, 
                            padding=3,
                            bias=False),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
        )
        
        self.conv2 = nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            stage(64,256,3, stride=1)
        )
        
        self.conv3 = stage(256,512,4,stride=2)
        self.conv4 = stage(512,1024,6,stride=2)
        self.conv5 = stage(1024,2048,3,stride=2) 
        self.fc = torch.nn.Linear(2048,self.out_channels)
        
    

    def forward(self,x):
        # print(f'resnet start: {x.shape}')
        # print('start of conv1')
        # print(f'input: {x.shape}')
        x = self.conv1(x)
        # print(f'output: {x.shape}')
        
        # print('\nstart of conv2')
        # print(f'input: {x.shape}')
        x = self.conv2(x)
        
        # print('\nstart of conv3')
        # print(f'input: {x.shape}')
        x = self.conv3(x)
        
        # print('\nstart of conv4')
        # print(f'input: {x.shape}')
        x = self.conv4(x)
        
        # print('\nstart of conv5')
        # print(f'input: {x.shape}')
        x = self.conv5(x)
        # print(f'resnet end: {x.shape}')
        
        # print('\nstart of classifier')
        # print(f'input: {x.shape}')
        x = torch.nn.AvgPool2d(kernel_size=(x.shape[2],x.shape[3]), stride=1, padding=0)(x)
        x = self.fc(torch.squeeze(x))
        # print(f'classifier end: {x.shape}')
        return x