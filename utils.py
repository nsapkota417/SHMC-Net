import torch
import torch.nn as nn
import torchvision
from skimage.io import imread, imsave, imshow
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.utils as vutils
import numpy as np
import os, random, subprocess, wandb, yaml
import yaml
import collections
from datetime import datetime
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.datasets as datasets
import torchvision.datasets as datasets
from pprint import pprint
import pandas as pd
from sklearn.metrics import accuracy_score

# from resnet50 import ResNet

class DotDict(dict):
    """ Dictionary that allows dot notation access (nested not supported). """
    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __getattr__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            raise AttributeError(item)
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result
    
def nested_dotdict(nested_dict):
    if not isinstance(nested_dict, dict):
        return nested_dict
    new_dict = {k: nested_dotdict(nested_dict[k]) for k in nested_dict}
    return DotDict(new_dict)

def get_top_n(epoch2metrics, metric_name='accuracy', top_n=10):
    top_met_ep = sorted([(v[metric_name], k) for k, v in epoch2metrics.items()],
                        reverse=True)[:top_n]
    bestmet2metep = {met_name: [epoch2metrics[ep[1]][met_name] for ep in top_met_ep] 
                     for met_name in epoch2metrics[next(iter(epoch2metrics.keys()))].keys()}
    bestmet2metep['Ep'] = [item[1] for item in top_met_ep]
    return bestmet2metep


def oversample_df(df, cnf):
    if not cnf.train.oversample:
        return df
    class_counts = df['label'].value_counts()
    max_count = class_counts.max()
    max_value = class_counts.idxmax()
    
    # Oversample minority classes
    oversampled_dfs = []
    for class_val, count in class_counts.items():
        if class_val != max_value:
            df_class = df[df['label'] == class_val]
            oversampled_df = df_class.sample(n=max_count, replace=True)
            oversampled_dfs.append(oversampled_df)

    # Concatenate oversampled minority class DataFrames with the majority class DataFrame
    oversampled_dfs.append(df[df['label'] == max_value])
    oversampled_df = pd.concat(oversampled_dfs)

    # Shuffle the oversampled DataFrame
    oversampled_df = oversampled_df.sample(frac=1).reset_index(drop=True)
    return oversampled_df
    
class SoftCELoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ce = torch.nn.CrossEntropyLoss()
        
    def forward(self, logits, targets, partial_labels):
        partial_labels = partial_labels.to(logits.device)
        targets = targets.to(logits.device)
        
        if self.config.train.soft_label == 'M':
            total_loss = self.ce(logits, partial_labels[:, 0]) + \
                         self.ce(logits, partial_labels[:, 1]) + \
                         self.ce(logits, partial_labels[:, 2])
            
            total_loss = total_loss / 3
        
        elif self.config.train.soft_label == 'W':

            alpha = self.config.train.sl_alpha
            total_loss = 0

            for idx in range(len(logits)):
                l0 = self.ce(logits[idx],partial_labels[idx,0])
                l1 = self.ce(logits[idx],partial_labels[idx,1])    
                l2 = self.ce(logits[idx],partial_labels[idx,2])                

                if partial_labels[idx,0] == partial_labels[idx,1]:
                    loss = alpha*(l0+l1)/2+(1-alpha)*l2

                elif partial_labels[idx,0] == partial_labels[idx,2]:
                    loss = alpha*(l0+l2)/2+(1-alpha)*l1

                elif partial_labels[idx,1] == partial_labels[idx,2]:
                    loss = alpha*(l1+l2)/2+(1-alpha)*l0

                total_loss = total_loss + loss

            total_loss = total_loss/len(logits)
            
        return total_loss

    
def mixup_data(x, y, partial_y, cnf):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    
    alpha = cnf.train.mixup_alpha
    mix_type = cnf.train.mixup_by
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
        
    
    
    if mix_type == 'B':
        index = torch.randperm(batch_size)
    else:
        index = []
        for idx in range(batch_size):
            cid = y[idx].item()
            mix_cand = [idx for idx in range(len(y)) if y[idx]==cid]
            idx = random.choice(mix_cand)
            index.append(idx)
        
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # y_a, y_b = y, y[index]
    
    if 'scian' in cnf.train.meta and not cnf.train.scian_total_agg:
        y1, y2, y3 = partial_y[0], partial_y[1], partial_y[2]
        y_1a, y_1b = y1, y1[index]
        y_2a, y_2b = y2, y2[index]
        y_3a, y_3b = y3, y3[index]

        return mixed_x, [y_1a, y_2a, y_3a], [y_1b,y_2b,y_3b], lam, index
    else:
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam, index

def minority_mixup_data(x, y, partial_y, cnf):
    '''
    input:
            x : images, b*c*h*w
            y : labels, b*1
            partial_y : partial labels, b*3
            alpha: alpha value for mixup
            minority class : list of class ids belonging to minority class
            
    Returns:
            mixed inputs : b*c*h*w, 
            pairs of targets : list of partial labels of sample and mixed image
                                b*3, b*3
            lambda : strength of mixup 
    '''
    alpha = cnf.train.mixup_alpha
    mix_type = cnf.train.mixup_by
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size()[0]
    index = torch.tensor(np.arange(0,batch_size)).long()
    cls_pos = [[] for _ in range(5)]

    for i, ci in enumerate(y):
        cls_pos[ci].append(i)
    
    if mix_type == 'C':
        for cid, pos_idxs in enumerate(cls_pos):
            if cid != 4:
                index[pos_idxs] = torch.tensor(np.random.permutation(pos_idxs)).long()
    else:
        flat_minority = [item for sublist in cls_pos[0:-1] for item in sublist]
        index[flat_minority] = torch.tensor(np.random.permutation(flat_minority)).long()
        
    mixed_x = lam * x + (1 - lam) * x[index, :]    
    y1, y2, y3 = partial_y[0], partial_y[1], partial_y[2]
    y_1a, y_1b = y1, y1[index]
    y_2a, y_2b = y2, y2[index]
    y_3a, y_3b = y3, y3[index]

    return mixed_x, [y_1a, y_2a, y_3a], [y_1b,y_2b,y_3b], lam, index





