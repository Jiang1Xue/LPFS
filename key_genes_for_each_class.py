"""
作者：jx
日期：2019-1-6
版本：1
文件名：key_genes_for_each_class.py
功能：统计LPFS获得的关键基因集合
"""

import pandas as pd
import numpy as np

def read_file(path):

    genename_df = pd.read_csv(path)
    genename = genename_df.as_matrix
    genename = genename[:, 1]

    return genename

def union(gene_set1, gene_set2):

    values = np.vstack(gene_set1, gene_set2)

    return values

def main():
    """
    主函数
    """
    # ========= Step 1. 读入数据class 1 的关键基因列表 ===========
    dir_1 = '/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/1/'
    list = os.listdir(dir_1)

    gene_set_1 = np.array[0]

    for i in range(len(list)):
        print(list[i])
        file_code = list[i]
        path = '/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/1/' + file_code
        gename = read_file(path)
        gene_set_1 = union(gene_set_1, gename)

    final_gene_set_1 = pd.value_counts(gene_set_1)
    final_gene_set_1 = np.array(final_gene_set_1)
    final_gene_set_1 = pd.DataFrame(data=final_gene_set_1)
    final_gene_set_1_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set1.csv')

    # ========= Step 2. 读入数据class 2 的关键基因列表 ===========
    dir_2 = '/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/2/'
    list = os.listdir(dir_2)

    gene_set_2 = np.array[0]

    for i in range(len(list)):
        print(list[i])
        file_code = list[i]
        path = '/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/2/' + file_code
        gename = read_file(path)
        gene_set_2 = union(gene_set_2, gename)

    final_gene_set_2 = pd.value_counts(gene_set_2)
    final_gene_set_2 = np.array(final_gene_set_2)
    final_gene_set_2 = pd.DataFrame(data=final_gene_set_2)
    final_gene_set_2_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set2.csv')

    # ========= Step 3. 读入数据class 3 的关键基因列表 ===========
    dir_3 = '/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/3/'
    list = os.listdir(dir_3)

    gene_set_3 = np.array[0]

    for i in range(len(list)):
        print(list[i])
        file_code = list[i]
        path = '/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/3/' + file_code
        gename = read_file(path)
        gene_set_3= union(gene_set_3, gename)

    final_gene_set_3 = pd.value_counts(gene_set_3)
    final_gene_set_3 = np.array(final_gene_set_3)
    final_gene_set_3 = pd.DataFrame(data=final_gene_set_3)
    final_gene_set_3_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set3.csv')

    # ========= Step 4. 读入数据class 4 的关键基因列表 ===========
    dir_4 = '/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/4/'
    list = os.listdir(dir_4)

    gene_set_4 = np.array[0]

    for i in range(len(list)):
        print(list[i])
        file_code = list[i]
        path = '/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/4/' + file_code
        gename = read_file(path)
        gene_set_4 = union(gene_set_4, gename)

    final_gene_set_4 = pd.value_counts(gene_set_4)
    final_gene_set_4= np.array(final_gene_set_4)
    final_gene_set_4 = pd.DataFrame(data=final_gene_set_4)
    final_gene_set_4_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set4.csv')

    # ========= Step 5. 读入数据class 5 的关键基因列表 ===========
    dir_5 = '/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/5/'
    list = os.listdir(dir_5)

    gene_set_5 = np.array[0]

    for i in range(len(list)):
        print(list[i])
        file_code = list[i]
        path = '/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/5/' + file_code
        gename = read_file(path)
        gene_set_5 = union(gene_set_5, gename)

    final_gene_set_5 = pd.value_counts(gene_set_5)
    final_gene_set_5 = np.array(final_gene_set_5)
    final_gene_set_5 = pd.DataFrame(data=final_gene_set_5)
    final_gene_set_5_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set5.csv')

    # ========= Step 6. 读入数据class 6 的关键基因列表 ===========
    dir_6 = '/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/6/'
    list = os.listdir(dir_6)

    gene_set_6 = np.array[0]

    for i in range(len(list)):
        print(list[i])
        file_code = list[i]
        path = '/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/6/' + file_code
        gename = read_file(path)
        gene_set_6 = union(gene_set_6, gename)

    final_gene_set_6 = pd.value_counts(gene_set_6)
    final_gene_set_6 = np.array(final_gene_set_6)
    final_gene_set_6 = pd.DataFrame(data=final_gene_set_6)
    final_gene_set_6_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set6.csv')


if __name__ == '__main__':
    main()