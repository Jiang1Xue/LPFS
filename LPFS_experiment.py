"""
作者：jx
日期：2018-11-10
版本：1
文件名：LPFS_experiment.py
功能：Label propagation based semi-supervised feature selection algorithm
"""

from sklearn.preprocessing import MinMaxScaler
import math
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def combined_time_series(str_2m_q20_df, str_6m_q20_df, str_10m_q20_df, str_2m_df, str_6m_df, str_10m_df):
    """
    将三个时间节点的数据进行合并
    :param str_2m_q20_df:
    :param str_6m_q20_df:
    :param str_10m_q20_df:
    :param str_2m_df:
    :param str_6m_df:
    :param str_10m_df:
    :return:
    """

    str_2m_q20_matrix = str_2m_q20_df.values
    str_6m_q20_matrix = str_6m_q20_df.values
    str_10m_q20_matrix = str_10m_q20_df.values
    str_2m_matrix = str_2m_df.values
    str_6m_matrix = str_6m_df.values
    str_10m_matrix = str_10m_df.values

    str_matrix = np.hstack((str_2m_q20_matrix, str_6m_q20_matrix[:, 1:]))
    str_matrix = np.hstack((str_matrix, str_10m_q20_matrix[:, 1:]))
    str_matrix = np.hstack((str_matrix, str_2m_matrix[:, 1:]))
    str_matrix = np.hstack((str_matrix, str_6m_matrix[:, 1:]))
    str_matrix = np.hstack((str_matrix, str_10m_matrix[:, 1:]))

    return str_matrix

def lpfs_algorithm(str_matrix, n):
    """
    基于标签传播聚类的半监督的特征选择筛选临床特征相关的基因
    :param str_matrix: 基因表达矩阵
    :param n: 类别数
    :return: cluster_indicator_matrix, cluster_indicator_ones_matrix, Feature_matrix, object, objective_1, objective_2
    """
    gename = str_matrix[:, 0]
    express_data = str_matrix[:, 1:]

    sample_express = express_data.T

    #min_max_scaler = MinMaxScaler()
    #sample_express = min_max_scaler.fit_transform(sample_express_)
    #express_data = sample_express.T

    r, c = sample_express.shape
    sample_num = r
    class_num = n
    gene_num = c

    # ============ Step 1. 算法的参数初始化 =============
    # 初始化类别标签矩阵
    Y = mat(zeros((sample_num, class_num)))
    Y[0:15, 0] = 1
    Y[16:23, 1] = 1
    Y[24:39, 2] = 1
    Y[40:47, 3] = 1
    Y[48:55, 4] = 1
    Y[56:63, 5] = 1

    #初始化特征选择矩阵
    Feature_matrix = mat(random.rand(gene_num, class_num))

    #初始化聚类指示矩阵
    cluster_indicator_matrix = Y
    cluster_indicator_ones_matrix = Y

    # 初始化参数miu， miu平衡在标签传播过程中节点之间的距离与其原始标签相似性的重要性
    miu = 0.2

    # 初始化参数beta，beta平衡在特征学习过程中稀疏约束的重要性
    beta = 0.2

    # 初始化参数s，筛选出的每个表型下受疾病严重影响的基因的个数
    s = 10

    # 初始化参数delta，确保网络中的权重在0到1之间
    delta = 200

    # ============= Step 1. 使用高斯核函数定义成对节点之间的关系，建立基因表达数据的无向图 ===========
    # 定义一个权重矩阵
    weight_matrix = mat(ones((r, r)))

    # 使用高斯核函数定义节点间的关系
    for i in range(r):
        print('网络中共有%d个节点，求解第%d个节点的连接关系'%(r, i))
        for j in range(r):
            tmp = 0
            for l in range(gene_num):
                tmp += power(sample_express[i, l] - sample_express[j, l], 2)
            tmp_ = tmp / (2*delta*delta)
            tmp_e = exp(-tmp_)
            weight_matrix[i, j] = tmp_e

    # 对权重矩阵进行正则化，即每行的元素除以该行元素的和, 以保证网络中的权重在0到1之间
    normalized_weight_matrix = weight_matrix
    d = []
    for i in range(r):
        tmp = sum(weight_matrix[i, :])
        d.append(tmp)
        print('网络中第%d个节点的加权连接值为%f' % (i, tmp))
        if d != 0:
            for j in range(r):
                normalized_weight_matrix[i, j] = weight_matrix[i, j]/tmp
    d = np.array(d).reshape(len(d))
    # print(normalized_weight_matrix)

    # ============ Step 2. 参数初始化 =========================================
    # 设置目标函数初始值为
    obj = 10000000
    obj_1_ = 10000000
    iter = 0
    object = []
    object_1 = []
    object_2 = []
    objective_2 = []

    while iter < 5:
        iter += 1
        obj_ = obj
        iter_2 = 0
        obj_2_ = 10000

        # ============ Step 2. 对目标函数进行求解 =========================================

        # =============== Step 2.2 求解特征选择矩阵 ==========================================
        while iter_2 < 5:
            iter_2 = iter_2 + 1

            # print('求解特征选择矩阵')
            # 构造辅助矩阵
            U = mat(zeros((class_num, class_num)))
            for j in range(class_num):
                tmp = 0
                for i in range(gene_num):
                    tmp = tmp + Feature_matrix[i, j] * Feature_matrix[i, j]
                U[j, j] = 1 / (2 * pow(tmp, 1 / 2))
                print(U[j, j])

            # 更新特征选择矩阵
            numerator_factor = np.dot(express_data, cluster_indicator_matrix)
            denominator_factor_1_1 = np.dot(express_data, sample_express)
            denominator_factor_1 = np.dot(denominator_factor_1_1, Feature_matrix)
            denominator_factor_2 = np.dot(Feature_matrix, U)
            denominator_factor = denominator_factor_1 + beta * denominator_factor_2

            for i in range(gene_num):
                # print('一共有%d个基因，计算到第%d个'%(gene_num, i))
                for j in range(class_num):
                    if denominator_factor[i, j] == 0:
                        Feature_matrix[i, j] = 0
                    else:
                        Feature_matrix[i, j] = Feature_matrix[i, j] * numerator_factor[i, j] / denominator_factor[i, j]
            # print(Feature_matrix)

            # 计算目标函数值
            # 计算目标函数的第三项
            inter_num3 = 0
            inter_num_1 = np.dot(sample_express, Feature_matrix)
            for i in range(sample_num):
                for j in range(class_num):
                    tmp = power(inter_num_1[i, j] - cluster_indicator_matrix[i, j], 2)
                    inter_num3 += tmp

            # 计算目标函数的第四项
            inter_num4 = 0
            for j in range(class_num):
                inter_num_1 = 0
                for i in range(gene_num):
                    tmp = power(Feature_matrix[i, j], 2)
                    inter_num_1 += tmp
                inter_num_2 = power(inter_num_1, 1 / 2)
                inter_num4 += inter_num_2

            # 计算目标函数值
            object2 = inter_num3 + beta * inter_num4
            # 计算两次迭代目标函数的差值
            dif_obj_2 = obj_2_ - object2
            obj_2_ = object2
            object_2.append(object2)
            print('求解特征选择矩阵,第 %d 次迭代目标函数值为：%f,两次迭代的差值为：%f' % (iter_2, object2, dif_obj_2))

        # ============= Step 2.1 求解聚类指示矩阵 =============================================
        # 求解聚类指示矩阵
        # print('求解聚类指示矩阵')
        I = np.eye(sample_num, dtype = int)
        inter_first_item = (1 + miu) * I - normalized_weight_matrix
        inter_second_item = inter_first_item.I
        inter_third_item = miu * Y + np.dot(sample_express, Feature_matrix)

        cluster_indicator_matrix = np.dot(inter_second_item, inter_third_item)
        # print(cluster_indicator_matrix)

        cluster_indicator_ones_matrix = mat(zeros((sample_num, class_num)))
        for i in range(sample_num):
            tmp = cluster_indicator_matrix[i, :].max()
            for j in range(class_num):
                if cluster_indicator_matrix[i, j] == tmp:
                    cluster_indicator_ones_matrix[i, j] = 1
                else:
                    cluster_indicator_ones_matrix[i, j] = 0

        # print(cluster_indicator_ones_matrix)
        # 求解目标函数值
        # 计算目标函数中的第一项
        inter_num1 = 0
        for i in range(sample_num):
            for j in range(sample_num):
                inter_num_1 = 0
                for l in range(class_num):
                    tmp = power(cluster_indicator_ones_matrix[i, l] / d[i] - cluster_indicator_ones_matrix[j, l] / d[j], 2)
                    inter_num_1 += tmp
                inter_num1 += weight_matrix[i, j] * inter_num_1

        # 计算目标函数中的第二项
        inter_num2 = 0
        for i in range(sample_num):
            for j in range(class_num):
                tmp = power(cluster_indicator_ones_matrix[i, j] - Y[i, j], 2)
                inter_num_1 += tmp
            inter_num2 += inter_num_1

        # 计算目标函数中的第三项
        inter_num3 = 0
        inter_num_1 = np.dot(sample_express, Feature_matrix)
        for i in range(sample_num):
            for j in range(class_num):
                tmp = power(inter_num_1[i, j] - cluster_indicator_matrix[i, j], 2)
            inter_num3 += tmp

        # 计算目标函数的值
        object1 = inter_num1 + miu * inter_num2 + inter_num3
        # 计算两次迭代目标函数的差值
        dif_obj_1 = obj_1_ - object1
        obj_1_ = object1
        object_1.append(object1)

        print('求解聚类矩阵, 第 %d 次迭代目标函数值为：%f, 两次迭代的差值为：%f' % (iter, object1, dif_obj_1))

        # =============== Step 2.3 计算目标函数值 ==========================================
        # 计算目标函数中的第一项
        inter_num1 = 0
        for i in range(sample_num):
            for j in range(sample_num):
                inter_num_1 = 0
                for l in range(class_num):
                    tmp = power(cluster_indicator_ones_matrix[i, l] / d[i] - cluster_indicator_ones_matrix[j, l] / d[j], 2)
                    inter_num_1 += tmp
                inter_num1 += weight_matrix[i, j] * inter_num_1

        # 计算目标函数中的第二项
        inter_num2 = 0
        for i in range(sample_num):
            for j in range(class_num):
                tmp = power(cluster_indicator_ones_matrix[i, j] - Y[i, j], 2)
                inter_num_1 += tmp
            inter_num2 += inter_num_1

        # 计算目标函数中的第三项
        inter_num3 = 0
        inter_num_1 = np.dot(sample_express, Feature_matrix)
        for i in range(sample_num):
            for j in range(class_num):
                tmp = power(inter_num_1[i, j] - cluster_indicator_matrix[i, j], 2)
            inter_num3 += tmp

        # 计算目标函数的第四项
        inter_num4 = 0
        for j in range(class_num):
            inter_num_1 = 0
            for i in range(gene_num):
                tmp = power(Feature_matrix[i, j], 2)
                inter_num_1 += tmp
            inter_num_2 = power(inter_num_1, 1 / 2)
            inter_num4 += inter_num_2

        # 计算目标函数的值
        obj = inter_num1 + miu * inter_num2 + inter_num3 + beta * inter_num4
        # 计算两次迭代目标函数的差值
        dif_obj = obj_ - obj
        tmp = [inter_num1, inter_num2, inter_num3, inter_num4, obj]
        object.append(tmp)

        objective_2.append(object_2)

        print('迭代次数为：%d,目标函数值为：%f,两次迭代的差值为：%f' % (iter, obj, dif_obj))

    object = np.array(object)
    objective_2 = np.array(objective_2)

    return cluster_indicator_matrix, cluster_indicator_ones_matrix, Feature_matrix, object, object_1, objective_2

def select_feature(str_matrix, Feature_matrix, n):
    """
    根据Rank-product计算Feature_matrix中每行元素的波动情况，筛选重要特征，并修改原始基因表达矩阵
    :param Feature_matrix: 特征选择矩阵
    :param str_matrix: 基因表达矩阵
    :param n: 聚类个数
    :return: modify_X, delete_gene, modify_gename
    """
    # =============== Step 1. 根据Rank-product计算Feature_matrix中每行元素的波动情况==========================
    #根据特征选择矩阵中每行元素的波动情况，按rank-product从大到小进行排序
    #高排名的行对应的原始数据中的列（即相应的特征）对样本所属类别具有较大的鉴别能力，应予以保留
    #低排名的行，表明该特征对于样本在不同类别的分类贡献相同，即对于样本的类别没有鉴别能力，属于冗余特征，应该删除
    #本实验将应该删除的特征列全部置0，从而更新原始数据

    gename = str_matrix[:, 0]
    express_data = str_matrix[:, 1:]
    sample_express = express_data.T

    # print('gename的长度为', len(gename))

    r, c = sample_express.shape
    sample_num = r
    class_num = n
    gene_num = c
    print('sample number为', r)
    print('gene number 为', c)

    rankp = []
    for i in range(gene_num):
        tmp = 0
        num = 0
        for j in range(class_num):
            for l in range(class_num):
                if j < l and Feature_matrix[i, j] != 0 and Feature_matrix[i, l] != 0:
                    num += 1
                    temp = Feature_matrix[i, j] / Feature_matrix[i, l]
                    if temp < 1:
                        temp = 1/temp
                    tmp += temp
        if num > 0:
            tmp = tmp / num
        rankp.append(tmp)

    rankp_series = pd.Series(rankp)
    rankp_rank = rankp_series.rank(ascending = False)
    rankp_rank_ = np.array(rankp_rank).reshape(len(rankp_rank), 1)

    rankp = np.array(rankp).reshape(len(rankp), 1)
    gename = np.array(gename).reshape(len(gename), 1)

    gene_rankp = np.hstack((gename, rankp))
    gene_rankp = np.hstack((gene_rankp, rankp_rank_))
    gene_rankp = np.array(gene_rankp)

    # 对rankp值按从大到小进行排序
    rankp_ = list(rankp[:, 0])
    rankp_sort = sorted(rankp_, reverse = True)
    rankp_sort_ = np.array(rankp_sort)

    # 图片可视化rankp的变化曲线
    # plt.figure(1)
    # plt.plot(rankp_sort_)
    # plt.xlabel('Ranking')
    # plt.ylabel('The value of RankProduct')
    # plt.grid(True)
    # plt.show()
    #
    # plt.figure(2)
    # plt.plot(rankp_sort_[1:25])
    # plt.xlabel('Ranking')
    # plt.ylabel('The value of RankProduct')
    # plt.grid(True)
    # plt.show()

    # ============== Step 2. 根据特征矩阵中的rankp值修改基因表达矩阵 =================
    # 删除低排名的1000个行对应的基因表达数据的特征列
    # 保存删除的基因
    # 保存剩余的基因

    modify_X = sample_express
    modify_gename = []
    delete_gene = []
    delete_no = []

    for i in range(gene_num):
        if gene_rankp[i, 2] > gene_num - 1000:
            delete_gene.append(gene_rankp[i, :])
            delete_no.append(i)
        else:
            modify_gename.append(gene_rankp[i, 0])

    modify_gename = np.array(modify_gename).reshape(len(modify_gename), 1)

    delete_gene = np.array(delete_gene)
    delete_no = np.array(delete_no).reshape(len(delete_no), 1)
    delete_gene = np.hstack((delete_gene, delete_no))

    modify_X_express = np.delete(modify_X, delete_no, 1)

    print("modify_X_express 的shape为", modify_X_express.shape)
    modify_gename_ = modify_gename.T

    modify_X_ = np.vstack((modify_gename_, modify_X_express))
    print("modify_X_ 的shape为", modify_X_.shape)

    # 返回修改后的基因表达数据，修改后的基因列表，基因排名表，删除的基因列表，目标函数值
    return gene_rankp, modify_X_, delete_gene, modify_gename


def select_gene(train, rankp):
    """
    根据基因列表，筛选训练集中的基因的排名
    :param train_gename:
    :param rankp:
    :return:
    """
    select_rankp = []

    for i in range(len(train)):
        for j in range(len(rankp)):
            if train[i, 0] == rankp[j, 0]:
                tmp = list(rankp[j, :])
                tmp.append(train[i, 1])
                select_rankp.append(tmp)
                break

    select_rankp_matrix = np.array(select_rankp)

    return select_rankp_matrix


def lpfs_pipeline(str_matrix, n):
    """
    对各基因型的数据进行标签传播聚类的半监督的特征选择
    :param str_matrix, n
    :return:
    """
    # 初始化类别标签矩阵
    r, c = str_matrix.shape
    sample_num = r
    class_num = n

    Y = mat(zeros((sample_num, class_num)))
    Y[0:15, 0] = 1
    Y[16:23, 1] = 1
    Y[24:39, 2] = 1
    Y[40:47, 3] = 1
    Y[48:55, 4] = 1
    Y[56:63, 5] = 1

    print("LPFS 第1轮求解:")
    cluster_indicator_matrix1, cluster_indicator_ones_matrix1, Feature_matrix1, _, _, _ = lpfs_algorithm(str_matrix, n)
    gene_rankp1, modify_X1, delete_gene1, modify_gename1 = select_feature(str_matrix, Feature_matrix1, n)

    print("LPFS 第2轮求解:")
    modify_X1_ = modify_X1.T
    cluster_indicator_matrix2, cluster_indicator_ones_matrix2, Feature_matrix2, _, _, _ = lpfs_algorithm(modify_X1_, n)
    gene_rankp2, modify_X2, delete_gene2, modify_gename2 = select_feature(modify_X1_, Feature_matrix2, n)

    print("LPFS 第3轮求解:")
    modify_X2_ = modify_X2.T
    cluster_indicator_matrix3, cluster_indicator_ones_matrix3, Feature_matrix3, _, _, _ = lpfs_algorithm(modify_X2_, n)
    gene_rankp3, modify_X3, delete_gene3, modify_gename3 = select_feature(modify_X2_, Feature_matrix3, n)

    print("LPFS 第4轮求解:")
    modify_X3_ = modify_X3.T
    cluster_indicator_matrix4, cluster_indicator_ones_matrix4, Feature_matrix4, _, _, _ = lpfs_algorithm(modify_X3_, n)
    gene_rankp4, modify_X4, delete_gene4, modify_gename4 = select_feature(modify_X3_, Feature_matrix4, n)

    print("LPFS 第5轮求解:")
    modify_X4_ = modify_X4.T
    cluster_indicator_matrix5, cluster_indicator_ones_matrix5, Feature_matrix5, _, _, _ = lpfs_algorithm(modify_X4_, n)
    gene_rankp5, modify_X5, delete_gene5, modify_gename5 = select_feature(modify_X4_, Feature_matrix5, n)

    return cluster_indicator_ones_matrix5, Y, gene_rankp5, Feature_matrix5, modify_gename5


def main():
    """
    主函数
    """
    # ========= Step 1. 读入三个时间点的数据===========
    # 读入striatum组织的数据，为数据框格式
    str_2m_q20_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_2m_Q20.csv')
    str_6m_q20_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_6m_Q20.csv')
    str_10m_q20_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_10m_Q20.csv')

    str_2m_q80_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_2m_Q80.csv')
    str_6m_q80_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_6m_Q80.csv')
    str_10m_q80_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_10m_Q80.csv')

    str_2m_q92_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_2m_Q92.csv')
    str_6m_q92_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_6m_Q92.csv')
    str_10m_q92_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_10m_Q92.csv')

    str_2m_q111_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_2m_Q111.csv')
    str_6m_q111_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_6m_Q111.csv')
    str_10m_q111_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_10m_Q111.csv')

    str_2m_q140_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_2m_Q140.csv')
    str_6m_q140_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_6m_Q140.csv')
    str_10m_q140_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_10m_Q140.csv')

    str_2m_q175_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_2m_Q175.csv')
    str_6m_q175_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_6m_Q175.csv')
    str_10m_q175_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/processed_data/str_10m_Q175.csv')


    # 对不同的基因型的三个时间点的数据分别进行合并
    str_q20_q80_matrix = combined_time_series(str_2m_q20_df, str_6m_q20_df, str_10m_q20_df, str_2m_q80_df, str_6m_q80_df, str_10m_q80_df)
    str_q20_q92_matrix = combined_time_series(str_2m_q20_df, str_6m_q20_df, str_10m_q20_df, str_2m_q92_df, str_6m_q92_df, str_10m_q92_df)
    str_q20_q111_matrix = combined_time_series(str_2m_q20_df, str_6m_q20_df, str_10m_q20_df, str_2m_q111_df, str_6m_q111_df, str_10m_q111_df)
    str_q20_q140_matrix = combined_time_series(str_2m_q20_df, str_2m_q20_df, str_10m_q20_df, str_2m_q140_df, str_6m_q140_df, str_10m_q140_df)
    str_q20_q175_matrix = combined_time_series(str_2m_q20_df, str_2m_q20_df, str_10m_q20_df, str_2m_q175_df, str_6m_q175_df, str_6m_q175_df)


    # ========= Step 2. 对各基因型的数据进行标签传播聚类的半监督的特征选择 ===============
    # 对q20_q80, q20_q80, q20_q80, q20_q80, q20_q80数据进行半监督的标签传播
    q2v8_indicator_ones_matrix, q2v8_Y, q2v8_gene_rankp, feature_matrix2v8, modify_gename2v8 = lpfs_pipeline(str_q20_q80_matrix, 6)

    #
    # # ========= Step 3. 从每次基因排名列表中，提取训练集中的基因的排名 ==============
    # 读入训练集中的数据
    train_hit_gene_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/data/trainhital.csv')
    train_hit_matrix = train_hit_gene_df.values

    train_hit_gename = train_hit_matrix[:, 0]
    train_hit_gename = np.array(train_hit_gename).reshape(len(train_hit_gename), 1)
    train_hit_labels = train_hit_matrix[:, 1]
    train_hit_labels = np.array(train_hit_labels).reshape(len(train_hit_labels), 1)

    train_nohit_gene_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/data/trainnohital.csv')
    train_nohit_matrix = train_nohit_gene_df.values

    train_nohit_gename = train_nohit_matrix[:, 0]
    train_nohit_gename = np.array(train_nohit_gename).reshape(len(train_nohit_gename), 1)
    train_nohit_labels = train_nohit_matrix[:, 1]
    train_nohit_labels = np.array(train_nohit_labels).reshape(len(train_nohit_labels), 1)

    train_gename = np.vstack((train_hit_gename, train_nohit_gename))
    train_labels = np.vstack((train_hit_labels, train_nohit_labels))
    train = np.hstack((train_gename, train_labels))
    #
    # # 提取训练集中的基因的排名
    train_rank_2v80 = select_gene(train, q2v8_gene_rankp)


    # # =======Step 4. 保存文件============
    # # 将Numpy.array格式转化为pandas.dataframe格式
    v8_indicator_ones_matrix_df = pd.DataFrame(data = q2v8_indicator_ones_matrix)
    v8_Feature_matrix_df = pd.DataFrame(data = feature_matrix2v8)
    v8_gene_rankp_df = pd.DataFrame(data = q2v8_gene_rankp)
    v8_modify_gename_df = pd.DataFrame(data = modify_gename2v8)
    v8_train_rank_df = pd.DataFrame(data = train_rank_2v80)

    v8_indicator_ones_matrix_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q80/1/indicator_matrix.csv')
    v8_Feature_matrix_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q80/1/feature_matrix.csv')
    v8_gene_rankp_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q80/1/gene_rankp.csv')
    v8_modify_gename_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q80/1/modify_gename.csv')
    v8_train_rank_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q80/1/train_rank.csv')

    q2v92_indicator_ones_matrix, q2v92_Y, q2v92_gene_rankp, feature_matrix2v92, modify_gename2v92 = lpfs_pipeline(
        str_q20_q92_matrix, 6)
    q2v111_indicator_ones_matrix, q2v111_Y, q2v111_gene_rankp, feature_matrix2v111, modify_gename2v111 = lpfs_pipeline(
        str_q20_q111_matrix, 6)
    q2v140_indicator_ones_matrix, q2v140_Y, q2v140_gene_rankp, feature_matrix2v140, modify_gename2v140 = lpfs_pipeline(
        str_q20_q140_matrix, 6)
    q2v175_indicator_ones_matrix, q2v175_Y, q2v175_gene_rankp, feature_matrix2v175, modify_gename2v175 = lpfs_pipeline(
        str_q20_q175_matrix, 6)

    train_rank_2v92 = select_gene(train, q2v92_gene_rankp)
    train_rank_2v111 = select_gene(train, q2v111_gene_rankp)
    train_rank_2v140 = select_gene(train, q2v140_gene_rankp)
    train_rank_2v175 = select_gene(train, q2v175_gene_rankp)

    v92_indicator_ones_matrix_df = pd.DataFrame(data = q2v92_indicator_ones_matrix)
    v92_Feature_matrix_df = pd.DataFrame(data = feature_matrix2v92)
    v92_gene_rankp_df = pd.DataFrame(data = q2v92_gene_rankp)
    v92_modify_gename_df = pd.DataFrame(data = modify_gename2v92)
    v92_train_rank_df = pd.DataFrame(data = train_rank_2v92)

    v111_indicator_ones_matrix_df = pd.DataFrame(data = q2v111_indicator_ones_matrix)
    v111_Feature_matrix_df = pd.DataFrame(data = feature_matrix2v111)
    v111_gene_rankp_df = pd.DataFrame(data = q2v111_gene_rankp)
    v111_modify_gename_df = pd.DataFrame(data = modify_gename2v111)
    v111_train_rank_df = pd.DataFrame(data = train_rank_2v111)

    v140_indicator_ones_matrix_df = pd.DataFrame(data = q2v140_indicator_ones_matrix)
    v140_Feature_matrix_df = pd.DataFrame(data = feature_matrix2v140)
    v140_gene_rankp_df = pd.DataFrame(data = q2v140_gene_rankp)
    v140_modify_gename_df = pd.DataFrame(data = modify_gename2v140)
    v140_train_rank_df = pd.DataFrame(data = train_rank_2v140)

    v175_indicator_ones_matrix_df = pd.DataFrame(data = q2v175_indicator_ones_matrix)
    v175_Feature_matrix_df = pd.DataFrame(data = feature_matrix2v175)
    v175_gene_rankp_df = pd.DataFrame(data = q2v175_gene_rankp)
    v175_modify_gename_df = pd.DataFrame(data = modify_gename2v175)
    v175_train_rank_df = pd.DataFrame(data = train_rank_2v175)

    # # 对文件进行输出

    v92_indicator_ones_matrix_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q92/1/indicator_matrix.csv')
    v92_Feature_matrix_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q92/1/feature_matrix.csv')
    v92_gene_rankp_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q92/1/gene_rankp.csv')
    v92_modify_gename_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q92/1/modify_gename.csv')
    v92_train_rank_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q92/1/train_rank.csv')

    v111_indicator_ones_matrix_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q111/1/indicator_matrix.csv')
    v111_Feature_matrix_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q111/1/feature_matrix.csv')
    v111_gene_rankp_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q111/1/gene_rankp.csv')
    v111_modify_gename_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q111/1/modify_gename.csv')
    v111_train_rank_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q111/1/train_rank.csv')

    v140_indicator_ones_matrix_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q140/1/indicator_matrix.csv')
    v140_Feature_matrix_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q140/1/feature_matrix.csv')
    v140_gene_rankp_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q140/1/gene_rankp.csv')
    v140_modify_gename_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q140/1/modify_gename.csv')
    v140_train_rank_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q140/1/train_rank.csv')

    v175_indicator_ones_matrix_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q175/1/indicator_matrix.csv')
    v175_Feature_matrix_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q175/1/feature_matrix.csv')
    v175_gene_rankp_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q175/1/gene_rankp.csv')
    v175_modify_gename_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q175/1/modify_gename.csv')
    v175_train_rank_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q175/1/train_rank.csv')

if __name__ == '__main__':
    main()