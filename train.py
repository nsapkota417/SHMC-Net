import torch
import numpy as np
import pandas as pd
import os, yaml, argparse, wandb, time, math
from datetime import datetime
from pprint import pprint
from torchsampler import ImbalancedDatasetSampler
from trainer import Trainer, train_forward
from utils import nested_dotdict, get_top_n, oversample_df, SoftCELoss
from sperm_dataset import classificationDataset
from model_select import classifierModel
run_start = time.time()
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import random

import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-cnf','--config', help='Description for foo argument', required=True)
args = parser.parse_args()




# read from config -------------------------------------------------------------
# config_path = 'bibm.yaml'
config_path = args.config  # 'bibm.yaml'
cnf = nested_dotdict(yaml.load(open(config_path,'r'),Loader = yaml.FullLoader))

seed = cnf.train.seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


epochs = cnf.train.epochs
model_save_path = cnf.train.model_save_path
batch_size = cnf.train.batch_size
crop_dim = cnf.train.crop_dim

train_mode = cnf.train.pretrained_weights.split('/')[-1][0:7]
fd = ''.join('x' if item == '-' else item[0] if item and item[0].isalpha() else '' for item in cnf.train.fuse_direction)

if cnf.train.in_channels in (2,4):
    run_name = (
            # f'bb.{cnf.train.backbone}_'
            # f'seed.{cnf.train.seed}_'
            # f'fd.{fd}_'
            f'mxBy.{cnf.train.mixup_by}'
            # f'{cnf.train.image_col[0:3]}I_'
            # f'{cnf.train.mask_col[0:3]}M_'
            # f'{cnf.train.in_channels}c_'
            # f'{train_mode}_'
            # f'ovrS.{str(cnf.train.oversample)[0]}_'
            # f'bs{cnf.train.batch_size}_'
            # f'ep{cnf.train.epochs}_'
            # f'dMsk.{str(cnf.train.dynamic_mask)[0]}_'
            # f'do{cnf.train.dropout}_'
            # f'wd{cnf.train.wd}_'
            # f"csv.{cnf.train.meta.split('_')[-1].split('.')[0]}_"
            # f'sftLb{cnf.train.sl_alpha}{cnf.train.soft_label}_'
            # f'gF.{str(cnf.train.gated_fusion)[0]}_'
            # f'mxUp.{str(cnf.train.mixup_aug)[0]}{cnf.train.mixup_alpha}_'
            # f'm2u.{cnf.train.metric_to_use}'
    )
else:
    run_name = (
            # f'bb.{cnf.train.backbone}_'
            # f'seed.{cnf.train.seed}_'
            f'{cnf.train.image_col[0:3]}Im_'
            f'{cnf.train.in_channels}ch_'
            # f'{cnf.train.aug_type}_'
            f'mxUp.{str(cnf.train.mixup_aug)[0]}{cnf.train.mixup_alpha}_'
            # f'{train_mode}'
            # f'bs{cnf.train.batch_size}_'
            # f'do{cnf.train.dropout}_'
            # f'wd{cnf.train.wd}_'
            # f"csv.{cnf.train.meta.split('_')[-1].split('.')[0]}_"
            # f'ovrSam.{cnf.train.oversample}')
    )
    
config_dict = {
        "run_name" : run_name,
        "meta" : cnf.train.meta,
        "epochs" : cnf.train.epochs,
        "batch_size" : cnf.train.batch_size,
        "lr" : cnf.train.lr,
        "wd" : cnf.train.wd,
        "dropout" : cnf.train.dropout,
        "backbone" : cnf.train.backbone,
        "pretrained_weights" : cnf.train.pretrained_weights,
        "in_channels" : cnf.train.in_channels,
        "oversample" : cnf.train.oversample, 
        "resize_model_input" : cnf.train.resize_model_input,
        "feature_merge" : cnf.train.feature_merge,
        "fuse_direction" : fd,
        "metric_to_use" : cnf.train.metric_to_use,
        "top_n" : cnf.train.top_n,
        "dynamic_masking" : cnf.train.dynamic_mask,
        "soft_label" : cnf.train.soft_label,
        "gated_fusion" : cnf.train.gated_fusion,
        "mixup"  : f'{cnf.train.mixup_aug}_{cnf.train.mixup_alpha}_{cnf.train.mixup_by}',
        "seed" : cnf.train.seed
    }

if cnf.train.project != '': 
    wandb.init(
        project = f'{cnf.train.project}',
        config = config_dict,
        name = run_name,
        notes = cnf.train.notes
    )
    
print("Training Configs:")
for key, value in config_dict.items():
    print(f'{key}: {value}')

print('--'*45)

gpu_idx = 0
if torch.cuda.is_available():
    gpu_idx = int(os.getenv('SGE_HGR_gpu_card'))
    print(f'Assigned GPU: {gpu_idx}')
else:
    print('Running on CPU')

avg_fold_scores_w = []
avg_fold_scores_m = []
avg_fold_met2use_w = []
avg_fold_met2use_m = []
avg_fold_scores_best = []
avg_fold_met2use_best = []
all_targets = []
all_preds = []
df1 = pd.read_csv(cnf.train.meta)

folds = ['F1', 'F2', 'F3', 'F4', 'F5']
for fold in folds:
    fold_start_time = time.time()
    
    # DATA ------------------------------------------------------------------------
    train_df = df1.loc[df1['fold'] != fold]
    test_df = df1.loc[df1['fold'] == fold]
    
    train_dataset = classificationDataset(dataFrame=train_df, 
                                          cnf = cnf,
                                          in_channels = cnf.train.in_channels,
                                          image_col=cnf.train.image_col,
                                          mask_col=cnf.train.mask_col,
                                          crop_dim=crop_dim,
                                          transform=False if cnf.train.aug_type == 'N' else True,
                                          aug_type = cnf.train.aug_type,
                                          is_test=False)
    
    test_dataset = classificationDataset(dataFrame=test_df, 
                                         cnf=cnf,
                                         in_channels = cnf.train.in_channels,
                                         image_col=cnf.train.image_col,
                                         mask_col=cnf.train.mask_col,
                                         crop_dim=crop_dim,
                                         transform=False,
                                         aug_type = cnf.train.aug_type,
                                         is_test=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=batch_size, 
                            shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size=batch_size, shuffle=False)
    # MODEL ------------------------------------------------------------------------
    model = classifierModel(cnf)

    if fold == 'F1':
        total_parameters = np.sum([p.numel() for p in model.parameters()])
        print(f'Backbone: {cnf.train.backbone}, # of parameters: {total_parameters}')
        print('--'*45)
        print('--'*45)

    print(f'FOLD {fold}:\n# train images: '
          f'{len(train_dataset)}, test images: {len(test_dataset)}\n')
    
    # TRAINING SET UP -------------------------------------------------------------
    if cnf.train.soft_label in ('W', 'M'):
        criterion = SoftCELoss(cnf)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cnf.train.lr)
    schedueler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                        T_max=epochs, eta_min=0,last_epoch=-1)

    trainer = Trainer(cnf=cnf,
                      model=model,
                      device='cuda' if torch.cuda.is_available() else 'cpu', 
                      device_idx=gpu_idx,
                      criterion=criterion,
                      optimizer=optimizer,
                      training_DataLoader=train_loader,
                      test_DataLoader=test_loader,
                      lr_scheduler=schedueler,
                      epochs=epochs,
                      epoch=0,
                      model_save_path=model_save_path)
    
    scores_dict_w, scores_dict_m, cm, error_logs, fold_preds, fold_targets, _ = train_forward(trainer, cnf = cnf, fold=fold)
    
    # get top_n from fold
    top_n_w = get_top_n(scores_dict_w, 
                      metric_name=cnf.train.metric_to_use, 
                      top_n=cnf.train.top_n)
    
    top_n_m = get_top_n(scores_dict_m, 
                      metric_name=cnf.train.metric_to_use, 
                      top_n=cnf.train.top_n)
    
    top_1_m = get_top_n(scores_dict_m, 
                      metric_name=cnf.train.metric_to_use, 
                      top_n=1)
    
    all_preds.append(fold_preds[top_1_m['Ep'][0]])
    all_targets.append(fold_targets)
    
    # append fold scores to global list
    avg_fold_scores_w.append(top_n_w)
    avg_fold_scores_m.append(top_n_m)
    avg_fold_scores_best.append(top_1_m)

    # average the scores
    avg_fold_w = {k: np.round(np.mean(v) if k!='Ep' else np.median(v), 2) for k, v in top_n_w.items()}
    avg_fold_m = {k: np.round(np.mean(v) if k!='Ep' else np.median(v), 2) for k, v in top_n_m.items()}

    avg_fold_met2use_m.append(avg_fold_m[cnf.train.metric_to_use])
    avg_fold_met2use_w.append(avg_fold_w[cnf.train.metric_to_use])
    avg_fold_met2use_best.append(top_1_m[cnf.train.metric_to_use])
    
    hours, rem = divmod(time.time()-fold_start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'\nFold: {fold}, time: {hours} h, {minutes} m \n')

    if cnf.train.show_scores_w:
        print(f'top_{cnf.train.top_n} scores by weighted avg:')
        pprint(top_n_w)
        print(f'\nAvg of top_{cnf.train.top_n} by weighted avg:')
        pprint(avg_fold_w)
    
    
    print(f'\ntop_{cnf.train.top_n} scores by macro avg:')
    pprint(top_n_m)
    print(f'\n\nAvg of top_{cnf.train.top_n} by macro avg:')
    pprint(avg_fold_m)
    
    if cnf.train.show_cm:
        for ep in top_n_m['Ep']:
            if ep in cm:
                print(f'\nCM for epoch {ep} of fold {fold}')
                pprint(cm[ep])
        print('--'*45)
        
    if cnf.train.show_error_logs:
        best_epoch = np.array(top_n_m['Ep']).max()
        if best_epoch in error_logs:
            print(f'\nerror logs for epoch {ep} of fold {fold}')
            pprint(error_logs[ep])
    print('--'*45)

final_score_w = {met: float(np.round(np.mean([top[met] for top in avg_fold_scores_w]),2)) for met in avg_fold_scores_w[0].keys()}
final_score_m = {met: float(np.round(np.mean([top[met] for top in avg_fold_scores_m]),2)) for met in avg_fold_scores_m[0].keys()}
final_score_top = {met: float(np.round(np.mean([top[met] for top in avg_fold_scores_best]),2)) for met in avg_fold_scores_best[0].keys()}

# compute globally instead of per fold:
flat_preds = np.concatenate(all_preds)
flat_targets = np.concatenate(all_targets)

accuracy_score = sum(1 for t, p in zip(flat_preds, flat_targets) if t == p) 
precision_global, recall_global, f1_global, _ = precision_recall_fscore_support(flat_targets, 
                                                               flat_preds,
                                                               beta=1.0,
                                                               average='macro',
                                                               zero_division=0)

p_g = np.round(precision_global*100, 2)
r_g = np.round(recall_global*100, 2)
mets_global_top1 = {}
mets_global_top1['#A'] = np.round(accuracy_score, 2)
mets_global_top1['A'] = np.round(accuracy_score/ len(flat_targets)*100, 2)
mets_global_top1['P'] = p_g
mets_global_top1['R'] = r_g
mets_global_top1['F1'] = np.round(f1_global*100, 2)


print(f'\n---------XX..XX Top {cnf.train.top_n} XX..XX---------')
print(f'\nAvg of top_{cnf.train.top_n} {cnf.train.metric_to_use} per fold by macro : {avg_fold_met2use_m}')
if cnf.train.show_scores_w:
    print(f'\nAvg of top_{cnf.train.top_n} per fold metrices by weighted average:')
    pprint(final_score_w)
print(f'\nAvg of top_{cnf.train.top_n} per fold metrices by macro average:')
pprint(final_score_m)
p, r = final_score_m['P'], final_score_m['R']
print(f"Final Score (harmonic mean of P & R) for F1 : {2*p*r/(p+r):.2f}%  ✅   ✅") 
print(f"Final Score Accuracy : {final_score_m['A']}%  ✅   ✅\n") 


# print top 1 results
print(f'\n---------XX..XX Top 1 XX..XX---------')
print(f'\nAvg of top_1 {cnf.train.metric_to_use} per fold by macro : {avg_fold_met2use_best}')
pprint(final_score_top)
p_t, r_t = final_score_top['P'], final_score_top['R']
print(f"Final Score (harmonic mean of P & R) for F1 : {2*p_t*r_t/(p_t+r_t):.2f}%  ✅   ✅") 
print(f"Final Score Accuracy : {final_score_top['A']}%  ✅   ✅ \n") 

# print top 1 results globally
print(f'\ntop_1 {cnf.train.metric_to_use} by macro computed Globally')
pprint(mets_global_top1)
print(f"Final Score (harmonic mean of P & R) for F1 : {2*p_g*r_g/(p_g+r_g):.2f}%  ✅   ✅\n") 
print(f"Final Score Accuracy : {mets_global_top1['A']}%  ✅   ✅") 

hours, rem = divmod(time.time()-run_start, 3600)
minutes, seconds = divmod(rem, 60)
print('--'*45)
print(f'\nEnd Of Run {run_name}\nBackbone: {cnf.train.backbone} , time: {hours} h, {minutes} m \n' )