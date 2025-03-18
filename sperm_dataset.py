import torch
import torchvision
import torchvision.datasets as Datasets
from torch.utils.data import Dataset
import numpy as np
import os, yaml, argparse, wandb
from datetime import datetime
from trainer import Trainer
from utils import nested_dotdict
import pandas as pd
from skimage.io import imread, imsave, imshow
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import random, math
import copy
from utils import oversample_df

# def display_image_pair(im1, im2):
#     f, axarr = plt.subplots(1,2)            
#     axarr[0].imshow(im1.squeeze(), cmap='gray')
#     axarr[1].imshow(im2.squeeze(), cmap='gray')
#     plt.show()

def display_image_pair(image1, image2):
    fig, axes = plt.subplots(1, 2)
    if image1.ndim == 3 and image1.shape[0] == 3:
        axes[0].imshow(image1.numpy().transpose(1, 2, 0))
    else:
        axes[0].imshow(image1.squeeze(), cmap='gray')
    axes[0].axis('off')
    
    if image2.ndim == 3 and image2.shape[0] == 3:
        axes[1].imshow(image2.numpy().transpose(1, 2, 0))
    else:
        axes[1].imshow(image2.squeeze(), cmap='gray')

    axes[1].axis('off')

    # Display the plot
    plt.show()


class classificationDataset(Dataset):
    def __init__(self, 
                 dataFrame,
                 cnf,
                 in_channels,
                 image_col,
                 mask_col,
                 aug_type,
                 crop_dim,
                 transform=None,
                 is_test=False,
                ):
        self.cnf = cnf
        self.origDataFrame = dataFrame
        self.dataFrame = oversample_df(dataFrame, self.cnf) if not is_test else dataFrame
        self.in_channels = in_channels
        self.image_col = image_col
        self.mask_col = mask_col
        self.aug_type = aug_type
        self.crop_dim = crop_dim
        self.transform = transform
        self.is_test = is_test
        
        # if 'mask' in self.image_col:
        #     assert self.in_channels in (1, 2)
    
    def __len__(self):
        return len(self.dataFrame)
    
    def __getitem__(self, idx):
        img_name = self.dataFrame.iloc[idx][self.image_col]
        image = imread(img_name) / 255
        if self.cnf.train.in_channels in (2,4):
            mask_name = self.dataFrame.iloc[idx][self.mask_col]
            mask = (imread(mask_name) / 255)
            mask[mask>0] = 1
            mask = mask.astype('uint8')
        # print(f'after loading, image: {image.shape}, mask: {mask.shape}')
        label = self.dataFrame.iloc[idx]['label']
        if 'scian' in self.cnf.train.meta and not self.cnf.train.scian_total_agg:
            partial_label = [self.dataFrame.iloc[idx]['label1'],
                             self.dataFrame.iloc[idx]['label2'],
                             self.dataFrame.iloc[idx]['label3']]
        else:
            partial_label = []
            
        if self.aug_type in ('A', 'SIM') and self.transform:
            if 'scian' in self.cnf.train.meta and label==3:
                aug_list = [                     
                    A.Rotate(limit=[10,10], p=0.5),
                    A.VerticalFlip(p=0.5), 
                    ToTensorV2(transpose_mask=False, always_apply=True)
                ]   
            elif 'smids' in self.cnf.train.meta:
                aug_list = [ 
                    A.Resize(self.crop_dim[0], self.crop_dim[0], p=1.0),
                    A.RandomSizedCrop(min_max_height=(45, 64), 
                                    height=self.crop_dim[0], 
                                    width=self.crop_dim[0], p=0.5),
                    A.Rotate(limit=[20,20], p=0.5),
                    A.VerticalFlip(p=0.5), 
                    A.ShiftScaleRotate(shift_limit=0.1, p=0.5),
                    ToTensorV2(transpose_mask=False, always_apply=True)
                ]
                
            else:
                aug_list = [ 
                    # A.Resize(self.crop_dim[0], self.crop_dim[0], p=1.0),
                    A.RandomSizedCrop(min_max_height=(45, 64), 
                                    height=self.crop_dim[0], 
                                    width=self.crop_dim[0], p=0.5),
                    A.Rotate(limit=[10,10], p=0.5),
                    A.VerticalFlip(p=0.5), 
                    A.ShiftScaleRotate(shift_limit=0.1, p=0.5),
                    ToTensorV2(transpose_mask=False, always_apply=True)
                ]
        else:
            aug_list = [
                A.Resize(self.crop_dim[0], self.crop_dim[0], p=1.0),
                ToTensorV2(transpose_mask=False, always_apply=True)
            ] 
            
        def convert_image(image, in_channels):
            in_channels = in_channels if in_channels % 2 != 0 else in_channels - 1
            if image.shape[0] == in_channels:
                return image
            elif image.shape[0] < in_channels:
                image = image.expand(3, -1, -1)
            else:
                image = image[0].unsqueeze(0)
            # assert image.shape[0] == in_channels
            return image
            
        
        if self.in_channels in (1, 3):
            assert self.aug_type != 'SIM'
            aug_im = convert_image(A.Compose(aug_list)(image=image)['image'], 
                                   self.in_channels)
            
        else:
            if self.aug_type == 'A':
                sample_aug = A.Compose(aug_list)(image=image, mask=mask)
                aug_im = convert_image(sample_aug['image'], self.in_channels)
                aug_mask = sample_aug['mask']
                # print(f'after aug, image: {aug_im.shape}, mask: {aug_mask.shape}')

            else:
                aug_im = convert_image(A.Compose(aug_list)(image=image)['image'], 
                                       self.in_channels)
                aug_mask = A.Compose(aug_list)(image=image, mask=mask)['mask']
                # print(f'after aug, image: {aug_im.shape}, mask: {aug_mask.shape}')
        
            if aug_mask.ndim == 2:
                aug_mask = aug_mask.unsqueeze(0)
            if self.cnf.train.dynamic_mask:
                aug_im = self.dynamic_mask(aug_im, aug_mask)
            
            # display_image_pair(aug_im, aug_mask)
            # import IPython; IPython.embed()
            aug_im = torch.cat((aug_im, aug_mask), dim=0)
            # print(f'after stacking, image: {aug_im.shape}')
        return aug_im, label, partial_label, img_name.split('/')[-1].split('.')[0]
    
    def shuffle_df(self):
        if not self.is_test and self.cnf.train.oversample:
            self.dataFrame = oversample_df(self.origDataFrame, self.cnf)
            
    
    def dynamic_mask(self, im, msk):
        '''
            input: array (1ch or 3ch)
            mask: array (1ch)
            
            returns: bg subtracted image wrt mask
        '''
        # import IPython; IPython.embed()
        dilated_mask = cv2.dilate(msk.numpy().transpose(1, 2, 0), 
                                  kernel=np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8), 
                                  iterations=random.randint(5,10))
        
        im_copy = copy.deepcopy(im.numpy().transpose(1, 2, 0))
        im_copy[dilated_mask==0] = im.max()
        
        return torch.tensor(im_copy.transpose(-1,0,1))
