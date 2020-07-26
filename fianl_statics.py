# -*- coding：utf-8 -*-
"""
作者：jx
日期：2020-07-24
版本：1
文件名：final_statics.py
功能：对结果进行统计
"""

import numpy as np
import pandas as pd
import os

q2v8_result1_df = pd.read_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q80_result1.csv')
q2v8_result1 = q2v8_result1_df.values
q2v8_result2_df = pd.read_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q80_result2.csv')
q2v8_result2 = q2v8_result2_df.values

q2v8_ham_los_mean = np.mean(q2v8_result1[:, 1])
q2v8_zero_one_los_mean = np.mean(q2v8_result1[:, 2])
q2v8_coverage_error_mean = np.mean(q2v8_result1[:, 3])
q2v8_auc_mean = np.mean(q2v8_result2[:, 1])
q2v8_aupr_mean = np.mean(q2v8_result2[:, 2])

q2v8_ham_los_std = np.std(q2v8_result1[:, 1])
q2v8_zero_one_los_std = np.std(q2v8_result1[:, 2])
q2v8_coverage_error_std = np.std(q2v8_result1[:, 3])
q2v8_auc_std = np.std(q2v8_result2[:, 1])
q2v8_aupr_std = np.std(q2v8_result2[:, 2])

print("q2v8_ham_los_mean:", q2v8_ham_los_mean)
print("q2v8_ham_los_std", q2v8_ham_los_std)
print("q2v8_zero_one_los_mean", q2v8_zero_one_los_mean)
print("q2v8_zero_one_los_std", q2v8_zero_one_los_std)
print("q2v8_coverage_error_mean", q2v8_coverage_error_mean)
print("q2v8_coverage_error_std", q2v8_coverage_error_std)
print("q2v8_auc_mean", q2v8_auc_mean)
print("q2v8_auc_std", q2v8_auc_std)
print("q2v8_aupr_mean", q2v8_aupr_mean)
print("q2v8_aupr_std", q2v8_aupr_std)


q2v92_result1_df = pd.read_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q92_result1.csv')
q2v92_result1 = q2v92_result1_df.values
q2v92_result2_df = pd.read_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q92_result2.csv')
q2v92_result2 = q2v92_result2_df.values

q2v92_ham_los_mean = np.mean(q2v92_result1[:, 1])
q2v92_zero_one_los_mean = np.mean(q2v92_result1[:, 2])
q2v92_coverage_error_mean = np.mean(q2v92_result1[:, 3])
q2v92_auc_mean = np.mean(q2v92_result2[:, 1])
q2v92_aupr_mean = np.mean(q2v92_result2[:, 2])

q2v92_ham_los_std = np.std(q2v92_result1[:, 1])
q2v92_zero_one_los_std = np.std(q2v92_result1[:, 2])
q2v92_coverage_error_std = np.std(q2v92_result1[:, 3])
q2v92_auc_std = np.std(q2v92_result2[:, 1])
q2v92_aupr_std = np.std(q2v92_result2[:, 2])

print("q2v92_ham_los_mean:", q2v92_ham_los_mean)
print("q2v92_ham_los_std", q2v92_ham_los_std)
print("q2v92_zero_one_los_mean", q2v92_zero_one_los_mean)
print("q2v92_zero_one_los_std", q2v92_zero_one_los_std)
print("q2v92_coverage_error_mean", q2v92_coverage_error_mean)
print("q2v92_coverage_error_std", q2v92_coverage_error_std)
print("q2v92_auc_mean", q2v92_auc_mean)
print("q2v92_auc_std", q2v92_auc_std)
print("q2v92_aupr_mean", q2v92_aupr_mean)
print("q2v92_aupr_std", q2v92_aupr_std)


q2v111_result1_df = pd.read_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q111_result1.csv')
q2v111_result1 = q2v111_result1_df.values
q2v111_result2_df = pd.read_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q111_result2.csv')
q2v111_result2 = q2v111_result2_df.values

q2v111_ham_los_mean = np.mean(q2v111_result1[:, 1])
q2v111_zero_one_los_mean = np.mean(q2v111_result1[:, 2])
q2v111_coverage_error_mean = np.mean(q2v111_result1[:, 3])
q2v111_auc_mean = np.mean(q2v111_result2[:, 1])
q2v111_aupr_mean = np.mean(q2v111_result2[:, 2])

q2v111_ham_los_std = np.std(q2v111_result1[:, 1])
q2v111_zero_one_los_std = np.std(q2v111_result1[:, 2])
q2v111_coverage_error_std = np.std(q2v111_result1[:, 3])
q2v111_auc_std = np.std(q2v111_result2[:, 1])
q2v111_aupr_std = np.std(q2v111_result2[:, 2])

print("q2v111_ham_los_mean:", q2v111_ham_los_mean)
print("q2v111_ham_los_std", q2v111_ham_los_std)
print("q2v111_zero_one_los_mean", q2v111_zero_one_los_mean)
print("q2v111_zero_one_los_std", q2v111_zero_one_los_std)
print("q2v111_coverage_error_mean", q2v111_coverage_error_mean)
print("q2v111_coverage_error_std", q2v111_coverage_error_std)
print("q2v111_auc_mean", q2v111_auc_mean)
print("q2v111_auc_std", q2v111_auc_std)
print("q2v111_aupr_mean", q2v111_aupr_mean)
print("q2v111_aupr_std", q2v111_aupr_std)

q2v140_result1_df = pd.read_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q140_result1.csv')
q2v140_result1 = q2v140_result1_df.values
q2v140_result2_df = pd.read_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q140_result1.csv')
q2v140_result2 = q2v140_result2_df.values

q2v140_ham_los_mean = np.mean(q2v140_result1[:, 1])
q2v140_zero_one_los_mean = np.mean(q2v140_result1[:, 2])
q2v140_coverage_error_mean = np.mean(q2v140_result1[:, 3])
q2v140_auc_mean = np.mean(q2v140_result2[:, 1])
q2v140_aupr_mean = np.mean(q2v140_result2[:, 2])

q2v140_ham_los_std = np.std(q2v140_result1[:, 1])
q2v140_zero_one_los_std = np.std(q2v140_result1[:, 2])
q2v140_coverage_error_std = np.std(q2v140_result1[:, 3])
q2v140_auc_std = np.std(q2v140_result2[:, 1])
q2v140_aupr_std = np.std(q2v140_result2[:, 2])

print("q2v140_ham_los_mean:", q2v140_ham_los_mean)
print("q2v140_ham_los_std", q2v140_ham_los_std)
print("q2v140_zero_one_los_mean", q2v140_zero_one_los_mean)
print("q2v140_zero_one_los_std", q2v140_zero_one_los_std)
print("q2v140_coverage_error_mean", q2v140_coverage_error_mean)
print("q2v140_coverage_error_std", q2v140_coverage_error_std)
print("q2v140_auc_mean", q2v140_auc_mean)
print("q2v140_auc_std", q2v140_auc_std)
print("q2v140_aupr_mean", q2v140_aupr_mean)
print("q2v140_aupr_std", q2v140_aupr_std)


q2v175_result1_df = pd.read_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q175_result1.csv')
q2v175_result1 = q2v175_result1_df.values
q2v175_result2_df = pd.read_csv('/lustre/home/acct-bmelgn/bmelgn-2/JX/LPFS/output/lpfs/q20_q175_result1.csv')
q2v175_result2 = q2v175_result2_df.values

q2v175_ham_los_mean = np.mean(q2v175_result1[:, 1])
q2v175_zero_one_los_mean = np.mean(q2v175_result1[:, 2])
q2v175_coverage_error_mean = np.mean(q2v175_result1[:, 3])
q2v175_auc_mean = np.mean(q2v175_result2[:, 1])
q2v175_aupr_mean = np.mean(q2v175_result2[:, 2])

q2v175_ham_los_std = np.std(q2v175_result1[:, 1])
q2v175_zero_one_los_std = np.std(q2v175_result1[:, 2])
q2v175_coverage_error_std = np.std(q2v175_result1[:, 3])
q2v175_auc_std = np.std(q2v175_result2[:, 1])
q2v175_aupr_std = np.std(q2v175_result2[:, 2])

print("q2v175_ham_los_mean:", q2v175_ham_los_mean)
print("q2v175_ham_los_std", q2v175_ham_los_std)
print("q2v175_zero_one_los_mean", q2v175_zero_one_los_mean)
print("q2v175_zero_one_los_std", q2v175_zero_one_los_std)
print("q2v175_coverage_error_mean", q2v175_coverage_error_mean)
print("q2v175_coverage_error_std", q2v175_coverage_error_std)
print("q2v175_auc_mean", q2v175_auc_mean)
print("q2v175_auc_std", q2v175_auc_std)
print("q2v175_aupr_mean", q2v175_aupr_mean)
print("q2v175_aupr_std", q2v175_aupr_std)
