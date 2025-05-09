import pickle
import json
import numpy as np
import os, sys
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
import biovec
from Bio.Seq import Seq
# from biovec import models
import warnings

warnings.filterwarnings("ignore")


def generate_Y(dataset):  # 生成的Y矩阵是（2183x1376）维用来标记是否激活抑制作用
    path = dataset + '/'
    affinity = open(path + 'affinity.tsv', 'r').readlines()
    dict_prot_thisSet = json.load(open(path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    dict_comp_thisSet = json.load(open(path + 'compounds.txt'), object_pairs_hook=OrderedDict)
    print("generate Y by dataset:", dataset)
    rows, cols = len(dict_comp_thisSet.keys()), len(dict_prot_thisSet.keys())
    print("effictive Drugs,proteins:", rows, cols)
    y = np.full((rows, cols), np.nan)
    aff_matrx = []
    for i in affinity:
        tmp_list = i.strip().split('\t')
        try:
            tmp_list[2] = int(tmp_list[2])
        except:
            print(tmp_list, 'format should belike: protein_key \t compound_key \t affinity')
            exit(0)
        aff_matrx.append(tmp_list)
    prot_col = [i for i in dict_prot_thisSet.keys()]
    comp_row = [i for i in dict_comp_thisSet.keys()]
    count = 0
    same_entry = []
    same_index = []
    print("create affinity matrix...")

    for ptr in range(len(aff_matrx)):
        col = prot_col.index(aff_matrx[ptr][0])
        row = comp_row.index(aff_matrx[ptr][1])
        if np.isnan(y[row][col]):
            count += 1
            y[row][col] = aff_matrx[ptr][2]
        else:
            same_index.append(ptr)
            same_entry.append([row, col])
            y[row][col] += aff_matrx[ptr][2]

    # For regression, calculate avg of duplicated data
    while same_entry:
        n = 2
        index = same_entry.pop()
        while index in same_entry:
            same_entry.remove(index)
            n += 1
        row = index[0]
        col = index[1]
        y[row][col] = int(y[row][col] / n)  # 这一步实际上是在对y中对应于多个相同索引的元素进行平均。
    print('writing to local file...')
    yyy = open(path + 'Y', 'wb')
    print(" dataset:", dataset, "finished; raw entries:", len(affinity), "entries:", count)
    pickle.dump(y, yyy)
    return count


def generate_fold(dataset_path):  # 去重后得到的数量为11065，生成从0到11065的序列
    valid_entries = generate_Y(dataset_path)
    valid_index = [i for i in range(valid_entries)]
    with open(dataset_path + '/valid_entries.txt', 'w') as f:
        print(valid_index, file=f)
    print(dataset_path, 'valid entries:', valid_entries)


def seq_to_kmers(seq, k=3):
    N = len(seq)
    return [seq[i:i + k] for i in range(N - k + 1)]

#
# dataset = ['toy_cls','BindingDB_cls','DrugAI','CYP_cls'][int(sys.argv[1])]
# embedding_type = ['onehot', 'bio2vec', 'tape'][int(sys.argv[2])]

dataset = 'DrugAI'
# dataset_path = "datasets/" + dataset + '/train'
dataset_path = "dataset/train"
generate_fold(dataset_path)

