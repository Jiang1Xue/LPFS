""""
作者：jx
日期：2018-11-1
版本：1
文件名：data_process.py
功能：对striatum、cortex、liver三个组织的数据分别进行预处理
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def pre_select_gene(str_2m_df, str_6m_df, str_10m_df, N):
    """
    预处理基因表达数据
    1.删除在所有样本中表达值全为0的基因
    2.对基因表达数据进行归一化处理
    3.计算每个基因在所有样本中的均值和方差，并按从大到小进行排序
    4.根据Rank-product选择高排名的5000个基因做下一步排序

    :param str_2m_df:
    :param str_6m_df:
    :param str_10m_df:
    :param N
    :return: pre_select_gene
    """
    # 预处理基因表达数据
    #==============Step 1. 将三个时间段的样本合并，删除在所有样本中表达值全为0的基因==============
    str_2m_Oridata_matrix = str_2m_df.as_matrix()
    str_6m_Oridata_matrix = str_6m_df.as_matrix()
    str_10m_Oridata_matrix = str_10m_df.as_matrix()

    gename = str_2m_Oridata_matrix[:, 0]
    str_2m_data_matrix = str_2m_Oridata_matrix[:, 1:]
    str_6m_data_matrix = str_6m_Oridata_matrix[:, 1:]
    str_10m_data_matrix = str_10m_Oridata_matrix[:, 1:]

    #将三个时间阶段的数据合并到一个矩阵中
    gene_express_matrix = np.hstack((str_2m_data_matrix, str_6m_data_matrix))
    gene_express_matrix = np.hstack((gene_express_matrix, str_10m_data_matrix))

    #删除在所有样本中表达值全为0的基因，即均值为0
    proc1_gene_express_list = []
    proc1_gename = []
    for i in range(len(gename)):
        if sum(gene_express_matrix[i,:]) != 0:
            proc1_gene_express_list.append(gene_express_matrix[i, :])
            proc1_gename.append(gename[i])

    print('Total gene number:', len(gename))
    print('The selected gene number after filtering out 0', len(proc1_gename))

    #===============Step 2. 计算每一个基因在所有样本中的方差和均值，并按从大到小进行排序=============
    proc2_gene_express_matrix = np.array(proc1_gene_express_list)
    proc_gene_express_var = []
    proc_gene_express_mean = []

    #对每一列数据进行归一化处理
    scaler = MinMaxScaler()
    gene_express_scaled = scaler.fit_transform(proc2_gene_express_matrix)

    #计算每个基因在不同样本下的方差，并按方差从大到小进行排名
    for i in range(len(proc1_gename)):
        proc_gene_express_var.append(gene_express_scaled[i, :].var())
        proc_gene_express_mean.append(gene_express_scaled[i, :].mean())

    #将list格式转化为数组格式，进而进行排序
    proc2_var = np.array(proc_gene_express_var)
    proc2_mean = np.array(proc_gene_express_mean)
    #按从大到小进行排序，并记录排名对应的索引
    #proc2_sort = abs(np.sort(-proc2_var))
    #proc2_mean = abs(np.sort(-proc2_mean))
    proc2_var_sort_index = np.argsort(-proc2_var)
    proc2_mean_sort_index = np.argsort(-proc2_mean)

    print(proc2_var_sort_index)
    print(proc2_mean_sort_index)
    #计算综合排名
    Rank_Product = []
    for i in range(len(proc1_gename)):
        a = proc2_var_sort_index[i]
        b = proc2_mean_sort_index[i]
        c = a * b
        Rank_Product.append(c)

    Rank_Product_sort_index = np.argsort(Rank_Product)

    # 作图，画出归一化后的所有基因的方差和均值变化的曲线图
    #plt.figure(1)
    #plt.plot(proc_var_sort[500:])
    #plt.xlabel('Ranking')
    #plt.ylabel('Variance')
    #plt.title('Variances of scaled gene expression')
    #plt.grid(True)
    #plt.show()

    #plt.figure(2)
    #plt.plot(proc_mean_sort[500:])
    #plt.xlabel('Ranking')
    #plt.ylabel('Mean')
    #plt.title('Means of scaled gene expression')
    #plt.grid(True)
    #plt.show()


    #===============Step 3. 筛选高排名的基因 ================
    #筛选排名前N的基因，并提取这些基因在各个时间点下的基因表达数据
    proc1_gename = np.array(proc1_gename)
    proc2_gename = []

    for i in range(len(proc1_gename)):
             if Rank_Product_sort_index[i] < N:
                 proc2_gename.append(proc1_gename[i])

    pre_select_gename = np.array(proc2_gename)
    return pre_select_gename

def union_gene_list(gene_list1, gene_list2):
    """
    整合两个基因列表中的基因
    :return: gene_list 整合的基因列表
    """
    gene_list = []
    for i in range(len(gene_list1)):
        if gene_list1[i] not in gene_list2:
            gene_list.append(gene_list1[i])

    for i in range(len(gene_list2)):
        gene_list.append(gene_list2[i])

    print('基因列表长度为:', len(gene_list))
    return gene_list

def overlap_gene(gene_list1, gene_list2):
    """
    统计两个基因集合中包含了多少疾病相关基因，多少非疾病相关基因
    :param gene_list1:
    :param gene_list2:
    :return: 重合的基因
    """

    overlap_gene = [l for l in gene_list1 if l in gene_list2]
    print('重合基因个数:', len(overlap_gene))

    return overlap_gene

def get_gene_express(gene_name, str_2m_df, str_6m_df, str_10m_df):
    """
    提取出gene_name的基因表达数据
    :param gene_name:
    :param str_2m_df, str_6m_df, str_10m_df:
    :return: str_2m_select_gene_express, str_6m_select_gene_express, str_10m_select_gene_express
    """
    str_2m_matrix = str_2m_df.as_matrix()
    str_6m_matrix = str_6m_df.as_matrix()
    str_10m_matrix = str_10m_df.as_matrix()

    str_2m_select_gexpress = []
    str_6m_select_gexpress = []
    str_10m_select_gexpress = []

    for i in range(len(gene_name)):
        for j in range(len(str_2m_matrix)):
            if gene_name[i] == str_2m_matrix[j, 0]:
                str_2m_select_gexpress.append(str_2m_matrix[j, :])
                str_6m_select_gexpress.append(str_6m_matrix[j, :])
                str_10m_select_gexpress.append(str_10m_matrix[j, :])
                break

    return str_2m_select_gexpress, str_6m_select_gexpress, str_10m_select_gexpress

def main():
    """
    主函数
    """
    # ========= Step 1. 读入数据，预筛选各组织基因 ===========
    #读入striatum组织的数据，为数据框格式
    striatum_2m_df = pd.read_csv('./data/hdinhd/Striatum/striatum_2m_FPKM.csv')
    striatum_6m_df = pd.read_csv('./data/hdinhd/Striatum/striatum_6m_FPKM.csv')
    striatum_10m_df = pd.read_csv('./data/hdinhd/Striatum/striatum_10m_FPKM.csv')

    #对Striatum组织的数据初步预筛选基因
    str_gename = pre_select_gene(striatum_2m_df, striatum_6m_df, striatum_10m_df, 5000)

    # ===== Step 2. 统计预筛选出的基因与训练集基因的交集，并将训练集中的基因加入整合基因列表=======
    #读入训练集中的疾病相关基因，并将其添加进入数据
    hit_gene_df = pd.read_csv('./data/trainhital.csv')
    hit_gene = hit_gene_df.as_matrix()
    hit_gename = hit_gene[:, 0]

    nohit_gene_df = pd.read_csv('./data/trainnohital.csv')
    nohit_gene = nohit_gene_df.as_matrix()
    nohit_gename = nohit_gene[:, 0]

    overlap_gene(str_gename, hit_gename)
    overlap_gene(str_gename, nohit_gename)

    train = union_gene_list(hit_gename, nohit_gename)
    final_gene_list = union_gene_list(str_gename, train)

    #========Step 3.根据最终基因列表，提取各组织基因表达数据并保存=========
    str_2m_gene_express, str_6m_gene_express, str_10m_gene_express = get_gene_express(final_gene_list, striatum_2m_df, striatum_6m_df, striatum_10m_df)


    # =======Step 4. 保存文件============
    # 将Numpy.array格式转化为pandas.dataframe格式
    final_gene_list_df = pd.DataFrame(data = final_gene_list)
    str_2m_gene_express_df = pd.DataFrame(data = str_2m_gene_express)
    str_6m_gene_express_df = pd.DataFrame(data = str_6m_gene_express)
    str_10m_gene_express_df = pd.DataFrame(data = str_10m_gene_express)

    # 对文件进行输出
    final_gene_list_df.to_csv('./output/pre_genename.csv')
    str_2m_gene_express_df.to_csv('./output/str_2m_gene_express.csv')
    str_6m_gene_express_df.to_csv('./output/str_6m_gene_express.csv')
    str_10m_gene_express_df.to_csv('./output/str_10m_gene_express.csv')

if __name__ == '__main__':
    main()