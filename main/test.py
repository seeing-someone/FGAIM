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
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
from models.FGAIM import FGAIM
from sklearn.metrics import *
from data_load import create_dataset_for_test
from utils import predicting,collate
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
num_feat_xp = 0
num_feat_xd = 0

model_name_seq = '_seq' if config.is_seq_in_graph is True else ''
model_name_con = '_con' if config.is_con_in_graph is True else ''
model_name_profile = '_pf' if config.is_profile_in_graph is True else ''
model_name_emb = '_emb' if config.is_emb_in_graph is True else ''

print('Using features: ')
print('Sequence.') if config.is_seq_in_graph else print('')
print('Contact.') if config.is_con_in_graph else print('')
print('SS + SA.') if config.is_profile_in_graph else print('')
print('Embedding.') if config.is_emb_in_graph else print('')

#dataset = config.dataset
dataset= '/public/home/tangyi/test/'
print('Dataset: ', dataset)


# model_st = modeling.__name__

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
LOG_INTERVAL = 20


test_data, num_feat_xp,num_feat_xd= create_dataset_for_test(dataset)
test_loader = DataLoader(test_data, batch_size=config.TEST_BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)

# training the model

#device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = modeling(num_features_xd=512,
                 num_features_xt=1024,
                 device=device).to(device)
# loss_fn = nn.MSELoss()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_mse = 1000
best_ci = 0
best_epoch = -1
#model_file_name = 'saved_model/' + setting[1:] + '/model_' + model_st + '_' + dataset \
                  # + model_name_emb + model_name_seq + model_name_con + model_name_profile \
                  # + setting + '.model'
model_file_name = '/public/home/tangyi/train_dtiam/saved_model/GEFA_SAGE_2025-04-02_09-15/2025-04-02_09-15_fold_1_best_model.pt'  # dataset +'_fold_5_best_model.pt'
# result_file_name = 'saved_model/' + setting[1:] + '/result_' + model_st + '_' + dataset \
#                    + model_name_emb + model_name_seq + model_name_con + model_name_profile \
#                    + setting + '.csv'
result_file_name = dataset +'saved_model.txt'

# new training
start_epoch = 0
early_stop = 100
best_auc = 0.5
stop_epoch = 0

for epoch in range(start_epoch, 1):

    current_time = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
    folder_path = os.path.join(dataset+'saved_model/',  f"{model_st}_{current_time}")
    os.makedirs(folder_path, exist_ok=True)  # exist_ok=True 可以防止文件夹已存在时抛出异常

    time_start = time.time()
    model.load_state_dict(torch.load( model_file_name, map_location=device), strict=False)
    T, S, test_loss = predicting(model, device, test_loader)
    # print(S.shape)
    # print(T.shape)
    S = S[:, 1]
    P = (S > 0.5).astype(int)
    AUROC = roc_auc_score(T, S)
    tpr, fpr, _ = precision_recall_curve(T, S)
    AUPR = auc(fpr, tpr)
    ACC = accuracy_score(T, P)
    REC = recall_score(T, P)
    f1 = f1_score(T, P)
    MCC = matthews_corrcoef(T, P)

    AUCS = [str(epoch + 1), str(format(time.time() - time_start, '.1f')),
            str(format(test_loss, '.4f')), str(format(AUROC, '.4f')), str(format(AUPR, '.4f')),
            str(format(ACC, '.4f')), str(format(REC, '.4f')), str(format(f1, '.4f')), str(format(MCC, '.4f'))]
    print('epoch\ttime\tval_loss\tAUROC\tAUPR\tACC\tRecall\tf1\tMCC')
    print('\t'.join(map(str, AUCS)))
    with open(result_file_name, 'w') as f:
        f.write(','.join(map(str, AUCS)))

        # Save metrics to a CSV file
        metrics_filename = os.path.join(folder_path, f'{current_time}_fold_1_best_metrics.csv')

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




