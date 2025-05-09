import copy
import os
import pandas as pd
import pickle
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric import data as DATA
from torch_geometric.data import Batch
import json
from graph_conversion import *
from metrics import *
from models.FGAIM import FGAIM
from utils import train,predicting,collate
from sklearn.model_selection import KFold
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import *
from data_load import create_dataset_for_train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_name_seq = '_seq' if config.is_seq_in_graph is True else ''
model_name_con = '_con' if config.is_con_in_graph is True else ''
model_name_profile = '_pf' if config.is_profile_in_graph is True else ''
model_name_emb = '_emb' if config.is_emb_in_graph is True else ''

print('Using features: ')
print('Sequence.') if config.is_seq_in_graph else print('')
print('Contact.') if config.is_con_in_graph else print('')
print('SS + SA.') if config.is_profile_in_graph else print('')
print('Embedding.') if config.is_emb_in_graph else print('')

# dataset = config.dataset
dataset = '/public/home/tangyi/train_dtiam/'
print('Dataset: ', dataset)


modeling = FGAIM
model_st = modeling.__name__

cuda_name = "cuda:" + str(config.cuda)
print('CUDA name:', cuda_name)

set_num = config.setting
settings = ['_setting_1', '_setting_2', '_setting_3', '_setting_4']
setting = settings[set_num]
print("Setting: ", setting)

print('Train batch size: ', config.TRAIN_BATCH_SIZE)
print('Test batch size: ', config.TEST_BATCH_SIZE)

LR = config.LR
print("Learning rate: ", LR)
print('Number of epoch: ', config.NUM_EPOCHS)
print(model_st)

import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, accuracy_score,
    recall_score, f1_score, matthews_corrcoef
)

def save_features_to_csv(features, labels, file_path):
    # 确保 features 是 CPU 上的 NumPy 数组
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()  # 转换到 CPU 并转为 NumPy
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()  # 同样转换标签
    print(features.shape)
    print(labels.shape)
    
    # 转换为 pandas DataFrame
    df = pd.DataFrame(features)
    df['Labels'] = labels  # 添加标签列

    # 保存到 CSV
    df.to_csv(file_path, index=False)

def train_model_with_cv(dataset, config):
    train_data, valid_data, num_feat_xp, num_feat_xd = create_dataset_for_train(dataset)
    data = list(train_data) + list(valid_data)  # Combine all data for cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 0
    # Get current time for file name


    current_time = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
    log_dir = dataset+'Training_Diary/'  # Target directory
    os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
    log_filename = os.path.join(log_dir, f'{current_time}_{model_st}_training_log.txt')

    folder_path = os.path.join(dataset+'saved_model/',  f"{model_st}_{current_time}")
    os.makedirs(folder_path, exist_ok=True)  # exist_ok=True 可以防止文件夹已存在时抛出异常


    # Open the file in write mode
    with open(log_filename, 'w') as log_file:

        for train_index, valid_index in kf.split(data):
            fold += 1
            log_file.write(f'Fold {fold}/{kf.get_n_splits()}\n')  # Log fold start
            print(f'Fold {fold}/{kf.get_n_splits()}')  # Print to console

            # Split data into training and validation sets
            train_dataset = torch.utils.data.Subset(data, train_index)
            valid_dataset = torch.utils.data.Subset(data, valid_index)

            train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True,
                                      collate_fn=collate, num_workers=0)
            valid_loader = DataLoader(valid_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=False,
                                      collate_fn=collate, num_workers=0)

            # Initialize model, optimizer, and other necessary variables
            device = 'cuda:0'  # Change to 'cuda:0' if using GPU
            model = modeling(num_features_xd=512, num_features_xt=1045, device=device).to(device)
            # model = modeling(num_features_xd=num_feat_xd, num_features_xt=num_feat_xp, device=device).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
            best_auc = 0.0
            early_stop = 50
            stop_epoch = 0

            # Training loop
            for epoch in range(config.NUM_EPOCHS):
                time_start = time.time()
                train_loss = train(model, device, train_loader, optimizer, epoch + 1)
                T, S, val_loss, test_fea = predicting(model, device, valid_loader)  # Use valid_loader for validation

                S = S[:, 1]
                P = (S > 0.5).astype(int)
                AUROC = roc_auc_score(T, S)
                tpr, fpr, _ = precision_recall_curve(T, S)
                AUPR = auc(fpr, tpr)
                ACC = accuracy_score(T, P)
                precision = precision_score(T, P)
                REC = recall_score(T, P)
                f1 = f1_score(T, P)
                MCC = matthews_corrcoef(T, P)

                AUCS = [str(epoch + 1), str(format(time.time() - time_start, '.1f')), str(format(train_loss, '.4f')),
                        str(format(val_loss, '.4f')), str(format(AUROC, '.4f')), str(format(AUPR, '.4f')),
                        str(format(ACC, '.4f')),  str(format(precision, '.4f')), str(format(REC, '.4f')), str(format(f1, '.4f')),
                        str(format(MCC, '.4f'))]

                # Log the metrics to file
                log_file.write('epoch\ttime\ttrain_loss\tval_loss\tAUROC\tAUPR\tACC\tprecision\tRecall\tf1\tMCC\n')
                log_file.write('\t'.join(map(str, AUCS)) + '\n')

                # Also print the metrics to console
                print('epoch\ttime\ttrain_loss\tval_loss\tAUROC\tAUPR\tACC\tprecision\tRecall\tf1\tMCC')
                print('\t'.join(map(str, AUCS)))

                if AUROC >= best_auc:
                    stop_epoch = 0
                    best_auc = AUROC
                    # torch.save(model.state_dict(), f'{dataset}_fold_{fold}_best_model.pt')
                    torch.save(model.state_dict(),  os.path.join(folder_path, f'{current_time}_fold_{fold}_best_model.pt'))

                    # Save metrics to a CSV file
                    metrics_filename = os.path.join(folder_path, f'{current_time}_fold_{fold}_best_metrics.csv')


                    # 验证集的高维特征和标签保存
                    val_features_filename = os.path.join(folder_path, f'{current_time}_fold_{fold}_val_features.csv')
                    save_features_to_csv(test_fea, T, val_features_filename)

                    # Convert variables to pandas DataFrame
                    max_len = max(len(T), len(S), len(P), len(tpr), len(fpr))  # 找出最长的列
                    data_dict = {
                        "T (True labels)": list(T) + [None] * (max_len - len(T)),
                        "S (Scores)": list(S) + [None] * (max_len - len(S)),
                        "P (Predictions)": list(P) + [None] * (max_len - len(P)),
                        "tpr (True positive rate)": list(tpr) + [None] * (max_len - len(tpr)),
                        "fpr (False positive rate)": list(fpr) + [None] * (max_len - len(fpr))
                    }
                    df = pd.DataFrame(data_dict)

                    # 第一行手动写入 AUROC 和 AUPR
                    with open(metrics_filename, 'w') as f:
                        f.write(f'AUROC,{AUROC}\n')
                        f.write(f'AUPR,{AUPR}\n')

                    # 追加 DataFrame 到 CSV（不写入列名）
                    df.to_csv(metrics_filename, mode='a', index=False)


                else:
                    stop_epoch += 1

                if stop_epoch == early_stop:
                    log_file.write(f'(EARLY STOP) No improvement since epoch {epoch + 1}\n')
                    print(f'(EARLY STOP) No improvement since epoch {epoch + 1}')
                    break

            log_file.write(f'Fold {fold} finished with best AUROC: {best_auc:.4f}\n')
            print(f'Fold {fold} finished with best AUROC: {best_auc:.4f}\n')


# Example of running the function
train_model_with_cv(dataset, config)
