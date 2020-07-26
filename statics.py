# -*- coding：utf-8 -*-
"""
作者：jx
日期：2020-07-24
版本：1
文件名：statics.py
功能：对LPFS的特征选择结果计算auroc，aupr,对标签分类结果计算hamming_loss, zero_one_loss, coverage_error
"""

import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import coverage_error

def true_label_matrix(sample_num, class_num):
    # 定义类别标签矩阵
    Y = mat(zeros((sample_num, class_num)))
    Y[0:15, 0] = 1
    Y[16:23, 1] = 1
    Y[24:39, 2] = 1
    Y[40:47, 3] = 1
    Y[48:55, 4] = 1
    Y[56:63, 5] = 1

    return Y


def compute_evaluation(true_matrix, predict_matrix):
    h = hamming_loss(true_matrix, predict_matrix)
    z = zero_one_loss(true_matrix, predict_matrix)
    c = coverage_error(true_matrix, predict_matrix)

    result = [h, z, c]

    return result

def compute_auc(true_label, predict_score):

    auc = roc_auc_score(true_label, predict_score)
    precision, recall, _thresholds = metrics.precision_recall_curve(true_label, predict_score)
    aupr = metrics.auc(recall, precision)

    result = [auc, aupr]

    return result

def read_data(path):

    indicator_matrix_path = path + 'indicator_matrix.csv'
    train_rank_path = path + 'train_rank.csv'
    indicator_matrix_df = pd.read_csv(indicator_matrix_path)
    indicator_matrix = indicator_matrix_df.values
    indicator_matrix = indicator_matrix[:, 1:]

    train_rank_df = pd.read_csv(train_rank_path)
    train_rank = train_rank_df.values
    true_label = train_rank[:, 4]
    predict_score = train_rank[:, 2]

    return indicator_matrix, true_label, predict_score

def main():
    # =========== Step 1. 读入q20_q80的结果 ==============
    dir = '/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q80/'
    list = os.listdir(dir)

    q2v80_result1 = []
    q2v80_result2 = []

    for i in range(len(list)):
        print(list[i])
        subject_code = list[i]
        path = '/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q80/' + subject_code
        predict_matrix, true_label, predict_score = read_data(path)
        true_matrix = true_label_matrix(64, 6)
        evaluation_result1 = compute_evaluation(true_matrix, predict_matrix)
        q2v80_result1.append(evaluation_result1)
        evaluation_result2 = compute_auc(true_label, predict_score)
        q2v80_result2.append(evaluation_result2)

    q2v80_result1 = np.array(q2v80_result1)
    q2v80_result2 = np.array(q2v80_result2)

    # 保存结果
    q2v80_result1_df = pd.DataFrame(data=q2v80_result1)
    q2v80_result1_df.to_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q80_result1.csv')
    q2v80_result2_df = pd.DataFrame(data=q2v80_result2)
    q2v80_result2_df.to_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q80_result2.csv')

    # =========== Step 2. 读入q20_q92的结果 ==============
    dir = '/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q92/'
    list = os.listdir(dir)

    q2v92_result1 = []
    q2v92_result2 = []

    for i in range(len(list)):
        print(list[i])
        subject_code = list[i]
        path = '/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q92/' + subject_code
        predict_matrix, true_label, predict_score = read_data(path)
        true_matrix = true_label_matrix(64, 6)
        evaluation_result1 = compute_evaluation(true_matrix, predict_matrix)
        q2v92_result1.append(evaluation_result1)
        evaluation_result2 = compute_auc(true_label, predict_score)
        q2v92_result2.append(evaluation_result2)

    q2v92_result1 = np.array(q2v92_result1)
    q2v92_result2 = np.array(q2v92_result2)

    # 保存结果
    q2v92_result1_df = pd.DataFrame(data=q2v92_result1)
    q2v92_result1_df.to_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q92_result1.csv')
    q2v92_result2_df = pd.DataFrame(data=q2v92_result2)
    q2v92_result2_df.to_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q92_result2.csv')

    # =========== Step 1. 读入q20_q111的结果 ==============
    dir = '/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q111/'
    list = os.listdir(dir)

    q2v111_result1 = []
    q2v111_result2 = []

    for i in range(len(list)):
        print(list[i])
        subject_code = list[i]
        path = '/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q111/' + subject_code
        predict_matrix, true_label, predict_score = read_data(path)
        true_matrix = true_label_matrix(64, 6)
        evaluation_result1 = compute_evaluation(true_matrix, predict_matrix)
        q2v111_result1.append(evaluation_result1)
        evaluation_result2 = compute_auc(true_label, predict_score)
        q2v111_result2.append(evaluation_result2)

    q2v111_result1 = np.array(q2v111_result1)
    q2v111_result2 = np.array(q2v111_result2)

    # 保存结果
    q2v111_result1_df = pd.DataFrame(data=q2v111_result1)
    q2v111_result1_df.to_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q111_result1.csv')
    q2v111_result2_df = pd.DataFrame(data=q2v111_result2)
    q2v111_result2_df.to_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q111_result2.csv')

    # =========== Step 1. 读入q20_q140的结果 ==============
    dir = '/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q140/'
    list = os.listdir(dir)

    q2v140_result1 = []
    q2v140_result2 = []

    for i in range(len(list)):
        print(list[i])
        subject_code = list[i]
        path = '/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q140/' + subject_code
        predict_matrix, true_label, predict_score = read_data(path)
        true_matrix = true_label_matrix(64, 6)
        evaluation_result1 = compute_evaluation(true_matrix, predict_matrix)
        q2v140_result1.append(evaluation_result1)
        evaluation_result2 = compute_auc(true_label, predict_score)
        q2v140_result2.append(evaluation_result2)

    q2v140_result1 = np.array(q2v140_result1)
    q2v140_result2 = np.array(q2v140_result2)

    # 保存结果
    q2v140_result1_df = pd.DataFrame(data=q2v140_result1)
    q2v140_result1_df.to_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q140_result1.csv')
    q2v140_result2_df = pd.DataFrame(data=q2v140_result2)
    q2v140_result2_df.to_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q140_result2.csv')

    # =========== Step 1. 读入q20_q175的结果 ==============
    dir = '/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q175/'
    list = os.listdir(dir)

    q2v175_result1 = []
    q2v175_result2 = []

    for i in range(len(list)):
        print(list[i])
        subject_code = list[i]
        path = '/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q175/' + subject_code
        predict_matrix, true_label, predict_score = read_data(path)
        true_matrix = true_label_matrix(64, 6)
        evaluation_result1 = compute_evaluation(true_matrix, predict_matrix)
        q2v175_result1.append(evaluation_result1)
        evaluation_result2 = compute_auc(true_label, predict_score)
        q2v175_result2.append(evaluation_result2)

    q2v175_result1 = np.array(q2v175_result1)
    q2v175_result2 = np.array(q2v175_result2)

    # 保存结果
    q2v175_result1_df = pd.DataFrame(data=q2v175_result1)
    q2v175_result1_df.to_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q175_result1.csv')
    q2v175_result2_df = pd.DataFrame(data=q2v175_result2)
    q2v175_result2_df.to_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q175_result2.csv')


if __name__ == '__main__':
    main()