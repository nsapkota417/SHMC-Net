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
import torchmetrics
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import math, random
from utils import mixup_data, minority_mixup_data
import pprint

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

class Trainer:
    def __init__(self,
                cnf,
                model: torch.nn.Module,
                device: torch.device,
                device_idx: int,
                criterion: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                training_DataLoader: torch.utils.data.Dataset,
                # validation_DataLoader: torch.utils.data.Dataset = None,
                test_DataLoader: torch.utils.data.Dataset,
                lr_scheduler: torch.optim.lr_scheduler = None,
                epochs: int = 100,
                epoch: int = 0,
                model_save_path : str = '',
                backbone: str= ''
                ):
        self.cnf = cnf
        self.device = device
        self.device_idx = device_idx
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.test_DataLoader= test_DataLoader
        # self.validation_DataLoader = validation_DataLoader
        self.epochs = epochs
        self.epoch = epoch
        self.model_save_path = model_save_path
        self.backbone = backbone

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        
    def get_gpu_memory_map(self):
        """Get the current gpu usage.
        Returns
        -------
        usage: dict
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
        """
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        
        return gpu_memory_map

    def mem(self, gpu_indices):
        """ Get primary GPU card memory usage. """	
        if not torch.cuda.is_available():
            return -1.
        mem_map = self.get_gpu_memory_map()
        
        if isinstance(gpu_indices, collections.abc.Sequence):
            prim_card_num = gpu_indices[0]
        else:
            prim_card_num = int(gpu_indices)
        return mem_map[prim_card_num] / 1000

    def get_model_parameter(self):
        total_parameters = 0
        for p in self.model.parameters():
            total_parameters += p.numel()
        return total_parameters

    def save_model(self, name):
        timestamp = datetime.now().strftime("%m%d%H%M")
        path = os.path.join(self.model_save_path,f'{name}.pth')
        if not os.path.isdir(self.model_save_path):
            os.makedirs(self.model_save_path)
        torch.save(self.model.state_dict(), path)
        print(f'\nSaved Model @ {path}\n')


    

    # @classmethod
    def train(self, epoch, show_run=True):
        # import IPython; IPython.embed()

        
        
        # set data/model/criterion
        criterion = self.criterion
        model = self.model.train()
        # model = model.to(self.device)
        # model.train()
        train_losses = []
        # training loop
        flops_total = []
        # watch.tic('iter')
        
        # dff = self.training_DataLoader.dataset.dataFrame
        # print('\n', f'{epoch}\n', dff[dff['label'] == 2]['id'].value_counts())
        for i, (input, target, partial_label, name) in enumerate(self.training_DataLoader):
            # import IPython; IPython.embed()
            if self.cnf.train.in_channels == 4:
                org_input = input[:, 0:3, :, :]
                org_msk = input[:, -1:, :, :]
            
            if len(self.cnf.train.resize_model_input) != 0:
                new_h = random.choice(self.cnf.train.resize_model_input)
                input = nn.functional.interpolate(input, 
                                                  size=(new_h, new_h), 
                                                  mode='bilinear', 
                                                  align_corners=False)
            # Mix up
            if self.cnf.train.mixup_aug:
                if self.cnf.train.mixup_by == 'M':
                    input, target_a, target_b, lam, m_index = minority_mixup_data(input, 
                                                                target,
                                                                partial_label, 
                                                                self.cnf)
                   
                else:
                     input, target_a, target_b, lam, m_index = mixup_data(input, 
                                                                target,
                                                                partial_label, 
                                                                self.cnf)
                    
        
            # plot
            
            # for index in range(input.shape[0]):
            #     if self.cnf.train.mixup_aug:
            #         mix_ims = input[:, 0:3, :, :]
            #         mix_msks = input[:, -1:, :, :]
            #         print(f'iter: {i}, lambda = {lam}')
            #         print(name[index], target[index])
            #         print(name[m_index[index]], target[m_index[index]] )
            #         display_image_pair(org_input[index],org_msk[index])
            #         display_image_pair(org_input[m_index[index]],org_msk[m_index[index]])
            #         display_image_pair(mix_ims[index],mix_msks[index])
            #         print('\n')
            #     else:
            #         display_image_pair(org_input[index],org_msk[index])

                    

            # actual training code 
            input = input.to(self.device, dtype=torch.float)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            out = model(input, i, 'train')
            
            if "scian" in self.cnf.train.meta and not self.cnf.train.scian_total_agg :
                if self.cnf.train.mixup_aug and self.cnf.train.soft_label not in ('N') :
                    loss_d = lam * criterion(out, 
                                             target.long(), 
                                             torch.stack(target_a).T) + \
                             (1-lam) * criterion(out, 
                                                 target.long(), 
                                                 torch.stack(target_b).T)
                
                else:
                    if self.cnf.train.soft_label in ('W', 'M') and "scian" in self.cnf.train.meta:
                        loss_d = criterion(out, target.long(), torch.stack(partial_label).T) 
                    else:
                        loss_d = criterion(out, target.long())
            else:
                if self.cnf.train.mixup_aug:
                    loss_d = lam*criterion(out, target_a.to(out.device)) + \
                                (1-lam)*criterion(out, target_b.to(out.device))
                else:
                    loss_d = criterion(out, target.long())

            loss = loss_d
            loss_item = loss.item()
            
            train_losses.append(loss_item) 
            
            loss.backward() 
           
            self.optimizer.step()
            # if show_run:
            #     if i+1==1 or (i/2) == 0 or i+1==len(self.training_DataLoader):
            #         print(f'iteration: '
            #               f'{i+1}/{len(self.training_DataLoader)}, ' 
            #               f'loss: {loss_item:.4f}, '
            #               f'usage: {self.mem(self.device_idx):.1f} GB')
            
        if self.cnf.train.oversample:
            self.training_DataLoader.dataset.shuffle_df()
        
        curr_lr = self.optimizer.param_groups[0]['lr']
        self.lr_scheduler.step()
        if show_run:
            print(f'\n✅ ✅  epoch: {epoch+1}/{self.epochs}, ' 
                            f'average loss: {np.mean(train_losses):.4f}, ' 
                            f'curr_lr = {curr_lr:.6f}, '
                            f'usage: {self.mem(self.device_idx):.2f} GB ✅ ✅')
            print('--'*45)
        return np.mean(train_losses)
    
    @torch.inference_mode()
    def test(self, cnf, epoch, show_run = True):
        model = self.model.eval() 
        accuracy_score = 0
        total = 0
        
        # TP, FP, FN, TN = 0
        all_targets = []
        all_preds = []
        all_names = []
        error_logs_list = []
        
        for i, (input, target, partial_preds, name) in enumerate(self.test_DataLoader):
            input = input.to(self.device, dtype=torch.float)
            target = target.to(self.device)
            out = model(input, i, 'test')
            cid = torch.argmax(out, dim=1)
            error_log = []
            for idx in range(input.shape[0]):
                log = {}
                if cid[idx] == target[idx]:
                    accuracy_score += 1
                else:
                    log['name'] = name[idx]
                    log['true_class'] = target[idx].item()
                    log['pred_class'] = cid[idx].item()
                    error_log.append(log)
                total += 1
                all_targets.append(target[idx].item())
                all_preds.append(cid[idx].item())
                all_names.append(name[idx])
            error_logs_list.append(error_log)
            
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        # print(all_targets)
        # print(all_preds)
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_targets, 
                                                                   all_preds,
                                                                   average='weighted',
                                                                   zero_division=0)
        precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(all_targets, 
                                                                   all_preds,
                                                                   beta=1.0,
                                                                   average='macro',
                                                                   zero_division=0)
        
        # import IPython; IPython.embed()
        cm = confusion_matrix(all_targets, all_preds)
        # pca, pca_avg = per_class_accuracy(cm)
        
        
        mets_w = {}
        mets_w['#A'] = np.round(accuracy_score, 2)
        mets_w['A'] = np.round(accuracy_score/total*100, 2)
        mets_w['P'] = np.round(precision_w*100, 2)
        mets_w['R'] = np.round(recall_w*100, 2)
        mets_w['F1'] = np.round(f1_w*100, 2)
        
        mets_m = {}
        mets_m['#A'] = mets_w['#A']
        mets_m['A'] = mets_w['A']
        mets_m['P'] = np.round(precision_m*100, 2)
        mets_m['R'] = np.round(recall_m*100, 2)
        mets_m['F1'] = np.round(f1_m*100, 2)
        
        if show_run:
            # if epoch+1 == 1 or (epoch+1)/10==0 or epoch+1==self.epochs:
            print(f'Test Epoch: {epoch+1}/{self.epochs},\n'
                f'{mets_m}⬅ ⬅')
            print('--'*45)
        
        return mets_w, mets_m, cm, error_logs_list, all_preds, all_targets, all_names
    
    

def train_forward(trainer, cnf, fold):
    '''
    args: 
        epochs: int; total epochs
        log_name: string or empty; project name for wandb logs
        
    returns:
        all_acc: dict; (epochs, test_acc) for each epochs
    '''
    
    scores_needed_w = {}  
    scores_needed_m = {}
    cm_needed = {}
    error_logs_needed = {}
    best_score = cnf.train.score_th
    preds_needed = {}
    for epoch in range(cnf.train.epochs):
        epoch_mets = {}
        
        # training
        if not cnf.train.test_mode:
            epoch_loss = trainer.train(epoch=epoch, show_run=cnf.train.show_train_run)
            epoch_mets[f'{cnf.train.backbone}_train_loss_{fold}'] = epoch_loss

        # validating
        test_mets_w, test_mets_m, cm, error_logs, all_preds, all_targets, all_names  = trainer.test(cnf=cnf, 
                                                               epoch=epoch, 
                                                               show_run=cnf.train.show_test_run)
        
        # epoch_mets[f'test_{cnf.train.metric_to_use}_w_fold{fold[-1]}'] = test_mets_w[cnf.train.metric_to_use]
        epoch_mets[f'{cnf.train.backbone}_test_{cnf.train.metric_to_use}_m_fold{fold[-1]}'] = test_mets_m[cnf.train.metric_to_use]

        # wandb logging
        if epoch > 0.1*cnf.train.epochs:
            scores_needed_w[epoch+1] = test_mets_w
            scores_needed_m[epoch+1] = test_mets_m
            cm_needed[epoch+1] = cm
            error_logs_needed[epoch+1] = error_logs
            preds_needed[epoch+1] = all_preds
        if cnf.train.project != '' : wandb.log(epoch_mets)
        
        # save model
        if cnf.train.save_model:
            epoch_f1 = test_mets_m['F1']
            name = f"{fold}_ep{epoch+1}_{str(epoch_f1).replace('.','')}"
            # if (epoch+1 >= epoch/2) and ((epoch+1)%save_freq == 0 or epoch+1==epochs):
            if epoch >= cnf.train.epochs/2 and epoch_f1 > best_score:
                trainer.save_model(name)
                print('--'*45)
                best_score = epoch_f1

        
    return scores_needed_w, scores_needed_m, cm_needed, error_logs_needed, preds_needed, all_targets, all_names