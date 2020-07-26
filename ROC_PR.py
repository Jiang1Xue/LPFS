# -*- coding: utf-8 -*-
"""
画ROC曲线及PR曲线
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


def main():
    """
    作图ROC曲线，PR曲线
    """
    #==========Step 1. 读入FC的实验结果数据=============
    str_fc_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/fc/train_fc_comprehensive_rank.csv')
    str_fc_matrix = str_fc_df.values
    str_fc_labels = np.array(str_fc_matrix[:, 2]).reshape(len(str_fc_matrix), 1)

    str_fc_tpr = []
    str_fc_fpr = []
    str_fc_precision = []
    str_fc_recall = []
    tp = 0
    fp = 0
    for i in range(len(str_fc_labels)):
        if str_fc_labels[i] > 0:
            tp += 1
        else:
            fp += 1
        tpr_ = tp/88
        fpr_ = fp/430
        precision_ = tp/(tp+fp)
        recall_ = tp/88
        str_fc_tpr.append(tpr_)
        str_fc_fpr.append(fpr_)
        str_fc_precision.append(precision_)
        str_fc_recall.append(recall_)

    str_fc_tpr = np.array(str_fc_tpr).reshape(len(str_fc_tpr), 1)
    str_fc_fpr = np.array(str_fc_fpr).reshape(len(str_fc_fpr), 1)
    str_fc_precision = np.array(str_fc_precision).reshape(len(str_fc_precision), 1)
    str_fc_recall = np.array(str_fc_recall).reshape(len(str_fc_recall), 1)

    str_fc_roc_auc = auc(str_fc_fpr, str_fc_tpr)
    str_fc_rp_auc = auc(str_fc_recall, str_fc_precision)

    # ==========Step 2. 读入T-test的实验结果数据=============
    str_t_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/ttest/train_t_comprehensive_rank.csv')
    str_t_matrix = str_t_df.values
    str_t_labels = np.array(str_t_matrix[:, 2]).reshape(len(str_t_matrix), 1)

    str_t_tpr = []
    str_t_fpr = []
    str_t_precision = []
    str_t_recall = []
    tp = 0
    fp = 0
    for i in range(len(str_t_labels)):
        if str_t_labels[i] > 0:
            tp += 1
        else:
            fp += 1
        tpr_ = tp / 88
        fpr_ = fp / 430
        precision_ = tp / (tp + fp)
        recall_ = tp / 88
        str_t_tpr.append(tpr_)
        str_t_fpr.append(fpr_)
        str_t_precision.append(precision_)
        str_t_recall.append(recall_)

    str_t_tpr = np.array(str_t_tpr).reshape(len(str_t_tpr), 1)
    str_t_fpr = np.array(str_t_fpr).reshape(len(str_t_fpr), 1)
    str_t_precision = np.array(str_t_precision).reshape(len(str_t_precision), 1)
    str_t_recall = np.array(str_t_recall).reshape(len(str_t_recall), 1)

    str_t_roc_auc = auc(str_t_fpr, str_t_tpr)
    str_t_rp_auc = auc(str_t_recall, str_t_precision)

    # ==========Step 3. 读入deseq2的实验结果数据=============
    str_deseq_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/deseq2/train_deseq2_rank.csv')
    str_deseq_matrix = str_deseq_df.values
    str_deseq_labels = np.array(str_deseq_matrix[:, 2]).reshape(len(str_deseq_matrix), 1)

    str_deseq_tpr = []
    str_deseq_fpr = []
    str_deseq_precision = []
    str_deseq_recall = []
    tp = 0
    fp = 0
    for i in range(len(str_deseq_labels)):
        if str_deseq_labels[i] > 0:
            tp += 1
        else:
            fp += 1
        tpr_ = tp / 88
        fpr_ = fp / 430
        precision_ = tp / (tp + fp)
        recall_ = tp / 88
        str_deseq_tpr.append(tpr_)
        str_deseq_fpr.append(fpr_)
        str_deseq_precision.append(precision_)
        str_deseq_recall.append(recall_)

    str_deseq_tpr = np.array(str_deseq_tpr).reshape(len(str_deseq_tpr), 1)
    str_deseq_fpr = np.array(str_deseq_fpr).reshape(len(str_deseq_fpr), 1)
    str_deseq_precision = np.array(str_deseq_precision).reshape(len(str_deseq_precision), 1)
    str_deseq_recall = np.array(str_deseq_recall).reshape(len(str_deseq_recall), 1)

    str_deseq_roc_auc = auc(str_deseq_fpr, str_deseq_tpr)
    str_deseq_rp_auc = auc(str_deseq_recall, str_deseq_precision)

    # ==========Step 4. 读入edgeR的实验结果数据=============
    str_edgeR_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/edgeR/train_edgeR_rank.csv')
    str_edgeR_matrix = str_edgeR_df.values
    str_edgeR_labels = np.array(str_edgeR_matrix[:, 2]).reshape(len(str_edgeR_matrix), 1)

    str_edgeR_tpr = []
    str_edgeR_fpr = []
    str_edgeR_precision = []
    str_edgeR_recall = []
    tp = 0
    fp = 0
    for i in range(len(str_edgeR_labels)):
        if str_edgeR_labels[i] > 0:
            tp += 1
        else:
            fp += 1
        tpr_ = tp / 88
        fpr_ = fp / 430
        precision_ = tp / (tp + fp)
        recall_ = tp / 88
        str_edgeR_tpr.append(tpr_)
        str_edgeR_fpr.append(fpr_)
        str_edgeR_precision.append(precision_)
        str_edgeR_recall.append(recall_)

    str_edgeR_tpr = np.array(str_edgeR_tpr).reshape(len(str_edgeR_tpr), 1)
    str_edgeR_fpr = np.array(str_edgeR_fpr).reshape(len(str_edgeR_fpr), 1)
    str_edgeR_precision = np.array(str_edgeR_precision).reshape(len(str_edgeR_precision), 1)
    str_edgeR_recall = np.array(str_edgeR_recall).reshape(len(str_edgeR_recall), 1)

    str_edgeR_roc_auc = auc(str_edgeR_fpr, str_edgeR_tpr)
    str_edgeR_rp_auc = auc(str_edgeR_recall, str_edgeR_precision)

    # ==========Step 5. 读入limma的实验结果数据=============
    str_limma_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/limma/train_limma_rank.csv')
    str_limma_matrix = str_limma_df.values
    str_limma_labels = np.array(str_limma_matrix[:, 2]).reshape(len(str_limma_matrix), 1)

    str_limma_tpr = []
    str_limma_fpr = []
    str_limma_precision = []
    str_limma_recall = []
    tp = 0
    fp = 0
    for i in range(len(str_limma_labels)):
        if str_limma_labels[i] > 0:
            tp += 1
        else:
            fp += 1
        tpr_ = tp / 88
        fpr_ = fp / 430
        precision_ = tp / (tp + fp)
        recall_ = tp / 88
        str_limma_tpr.append(tpr_)
        str_limma_fpr.append(fpr_)
        str_limma_precision.append(precision_)
        str_limma_recall.append(recall_)

    str_limma_tpr = np.array(str_limma_tpr).reshape(len(str_limma_tpr), 1)
    str_limma_fpr = np.array(str_limma_fpr).reshape(len(str_limma_fpr), 1)
    str_limma_precision = np.array(str_limma_precision).reshape(len(str_limma_precision), 1)
    str_limma_recall = np.array(str_limma_recall).reshape(len(str_limma_recall), 1)

    str_limma_roc_auc = auc(str_limma_fpr, str_limma_tpr)
    str_limma_rp_auc = auc(str_limma_recall, str_limma_precision)

    # ==========Step 6. 读入fnmf的实验结果数据=============
    str_fnmf_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/fnmf/train_fnmf_rank.csv')
    str_fnmf_matrix = str_fnmf_df.values
    str_fnmf_labels = np.array(str_fnmf_matrix[:, 2]).reshape(len(str_fnmf_matrix), 1)

    str_fnmf_tpr = []
    str_fnmf_fpr = []
    str_fnmf_precision = []
    str_fnmf_recall = []
    tp = 0
    fp = 0
    for i in range(len(str_fnmf_labels)):
        if str_fnmf_labels[i] > 0:
            tp += 1
        else:
            fp += 1
        tpr_ = tp / 88
        fpr_ = fp / 430
        precision_ = tp / (tp + fp)
        recall_ = tp / 88
        str_fnmf_tpr.append(tpr_)
        str_fnmf_fpr.append(fpr_)
        str_fnmf_precision.append(precision_)
        str_fnmf_recall.append(recall_)

    str_fnmf_tpr = np.array(str_fnmf_tpr).reshape(len(str_fnmf_tpr), 1)
    str_fnmf_fpr = np.array(str_fnmf_fpr).reshape(len(str_fnmf_fpr), 1)
    str_fnmf_precision = np.array(str_fnmf_precision).reshape(len(str_fnmf_precision), 1)
    str_fnmf_recall = np.array(str_fnmf_recall).reshape(len(str_fnmf_recall), 1)

    str_fnmf_roc_auc = auc(str_fnmf_fpr, str_fnmf_tpr)
    str_fnmf_rp_auc = auc(str_fnmf_recall, str_fnmf_precision)

    # ==========Step 7. 读入jnmfma的实验结果数据=============
    str_jnmfma_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/jnmfma/train_jnmfma_rank.csv')
    str_jnmfma_matrix = str_jnmfma_df.values
    str_jnmfma_labels = np.array(str_jnmfma_matrix[:, 2]).reshape(len(str_jnmfma_matrix), 1)

    str_jnmfma_tpr = []
    str_jnmfma_fpr = []
    str_jnmfma_precision = []
    str_jnmfma_recall = []
    tp = 0
    fp = 0
    for i in range(len(str_jnmfma_labels)):
        if str_jnmfma_labels[i] > 0:
            tp += 1
        else:
            fp += 1
        tpr_ = tp / 88
        fpr_ = fp / 430
        precision_ = tp / (tp + fp)
        recall_ = tp / 88
        str_jnmfma_tpr.append(tpr_)
        str_jnmfma_fpr.append(fpr_)
        str_jnmfma_precision.append(precision_)
        str_jnmfma_recall.append(recall_)

    str_jnmfma_tpr = np.array(str_jnmfma_tpr).reshape(len(str_jnmfma_tpr), 1)
    str_jnmfma_fpr = np.array(str_jnmfma_fpr).reshape(len(str_jnmfma_fpr), 1)
    str_jnmfma_precision = np.array(str_jnmfma_precision).reshape(len(str_jnmfma_precision), 1)
    str_jnmfma_recall = np.array(str_jnmfma_recall).reshape(len(str_jnmfma_recall), 1)

    str_jnmfma_roc_auc = auc(str_jnmfma_fpr, str_jnmfma_tpr)
    str_jnmfma_rp_auc = auc(str_jnmfma_recall, str_jnmfma_precision)

    # ==========Step 8. 读入lpfs的实验结果数据=============
    #这里读入的是q20_q80的第10组实验结果
    lpfs_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/train_rank.csv')
    lpfs_matrix = lpfs_df.values
    lpfs_labels = np.array(lpfs_matrix[:, 4]).reshape(len(lpfs_matrix), 1)

    # 计算标签1的个数和标签-1的个数
    label_1 = 0
    label_1_ = 0
    for i in range(len(lpfs_labels)):
        if lpfs_labels[i] > 0:
            label_1 += 1
        else:
            label_1_ += 1
    lpfs_tpr = []
    lpfs_fpr = []
    lpfs_precision = []
    lpfs_recall = []
    tp = 0
    fp = 0
    for i in range(len(lpfs_labels)):
        if lpfs_labels[i] > 0:
            tp += 1
        else:
            fp += 1
        tpr_ = tp / label_1
        fpr_ = fp / label_1_
        precision_ = tp / (tp + fp)
        recall_ = tp / label_1
        lpfs_tpr.append(tpr_)
        lpfs_fpr.append(fpr_)
        lpfs_precision.append(precision_)
        lpfs_recall.append(recall_)

    lpfs_tpr = np.array(lpfs_tpr).reshape(len(lpfs_tpr), 1)
    lpfs_fpr = np.array(lpfs_fpr).reshape(len(lpfs_fpr), 1)
    lpfs_precision = np.array(lpfs_precision).reshape(len(lpfs_precision), 1)
    lpfs_recall = np.array(lpfs_recall).reshape(len(lpfs_recall), 1)

    lpfs_roc_auc = auc(lpfs_fpr, lpfs_tpr)
    lpfs_rp_auc = auc(lpfs_recall, lpfs_precision)

    # ==========Step 3. 画ROC曲线=============
    #画ROC图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来

    plt.figure()
    plt.plot(str_fc_fpr, str_fc_tpr, lw=1, label='FC, AUC=%0.3f' % (str_fc_roc_auc))
    plt.plot(str_t_fpr, str_t_tpr, lw=1, label='t-test, AUC=%0.3f'%(str_t_roc_auc))
    plt.plot(str_deseq_fpr, str_deseq_tpr, lw=1, label='DESeq2, AUC=%0.3f'%(str_deseq_roc_auc))
    plt.plot(str_edgeR_fpr, str_edgeR_tpr, lw=1, label='edgeR, AUC=%0.3f'%(str_edgeR_roc_auc))
    plt.plot(str_limma_fpr, str_limma_tpr, lw=1, label='limma, AUC=%0.3f'%(str_limma_roc_auc))
    plt.plot(str_fnmf_fpr, str_fnmf_tpr, lw=1, label='jNMFMA, AUC=%0.3f'%(str_fnmf_roc_auc))
    plt.plot(str_jnmfma_fpr, str_jnmfma_tpr, lw=1, label='FNMF, AUC=%0.3f'%(str_jnmfma_roc_auc))
    plt.plot(lpfs_fpr, lpfs_tpr, lw=1.5, label='LPFS, AUC=%0.3f' % (lpfs_roc_auc))
    # 画对角线
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label = 'Luck')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

    # ==========Step 4. 画PR曲线==============
    plt.figure()
    plt.plot(str_fc_recall, str_fc_precision, lw=1, label='FC, AUPR = %0.3f' % (str_fc_rp_auc))
    plt.plot(str_t_recall, str_t_precision, lw=1, label='t-test, AUPR = %0.3f' % (str_t_rp_auc))
    plt.plot(str_deseq_recall, str_deseq_precision, lw=1, label='DESeq2, AUPR=%0.3f'%(str_deseq_rp_auc))
    plt.plot(str_edgeR_recall, str_edgeR_precision, lw=1, label='edgeR, AUPR=%0.3f'%(str_edgeR_rp_auc))
    plt.plot(str_limma_recall, str_limma_precision, lw=1, label='limma, AUPR=%0.3f'%(str_limma_rp_auc))
    plt.plot(str_fnmf_recall, str_fnmf_precision, lw=1, label='jNMFMA, AUPR=%0.3f'%(str_fnmf_rp_auc))
    plt.plot(str_jnmfma_recall, str_jnmfma_precision, lw=1, label='FNMF, AUPR=%0.3f'%(str_jnmfma_rp_auc))
    plt.plot(lpfs_recall, lpfs_precision, lw=1.5, label='LPFS, AUPR=%0.3f' % (lpfs_rp_auc))

    # 画对角线
    plt.plot([0, 1], [0.17, 0.17], '--', color=(0.6, 0.6, 0.6), label='Luck')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision recall curve')
    plt.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    main()