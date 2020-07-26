"""
作者：jx
日期：2018-12-12
版本：1
文件名：keygene_rank.py
功能：根据特征选择矩阵筛选每个类别下的关键基因
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def key_gene_rank(gename_df, N, feature_matrix_df):
    gename_matrix = gename_df.values
    gename = gename_matrix[:, 1]
    gename = np.array(gename)
    gename = np.array(gename).reshape(len(gename), 1)

    feature_matrix = feature_matrix_df.values

    #为了保证rank()是按照矩阵中的元素的从大到小进行排名，对矩阵中的元素取负号
    class1_rank_series = pd.Series(-feature_matrix[:, 1])
    class1_rank_ = class1_rank_series.rank()
    class1_rank = np.array(class1_rank_).reshape(len(class1_rank_), 1)
    gename_rank_1 = np.hstack((gename, class1_rank))
    gename_rank_1 = np.array(gename_rank_1).reshape(len(gename), 2)

    class2_rank_series = pd.Series(-feature_matrix[:, 2])
    class2_rank_ = class2_rank_series.rank()
    class2_rank = np.array(class2_rank_).reshape(len(class2_rank_), 1)
    gename_rank_2 = np.hstack((gename, class2_rank))
    gename_rank_2 = np.array(gename_rank_2).reshape(len(gename), 2)

    class3_rank_series = pd.Series(-feature_matrix[:, 3])
    class3_rank_ = class3_rank_series.rank()
    class3_rank = np.array(class3_rank_).reshape(len(class3_rank_), 1)
    gename_rank_3 = np.hstack((gename, class3_rank))
    gename_rank_3 = np.array(gename_rank_3).reshape(len(gename), 2)

    class4_rank_series = pd.Series(-feature_matrix[:, 4])
    class4_rank_ = class4_rank_series.rank()
    class4_rank = np.array(class4_rank_).reshape(len(class4_rank_), 1)
    gename_rank_4 = np.hstack((gename, class4_rank))
    gename_rank_4 = np.array(gename_rank_4).reshape(len(gename), 2)

    class5_rank_series = pd.Series(-feature_matrix[:, 5])
    class5_rank_ = class5_rank_series.rank()
    class5_rank = np.array(class5_rank_).reshape(len(class5_rank_), 1)
    gename_rank_5 = np.hstack((gename, class5_rank))
    gename_rank_5 = np.array(gename_rank_5).reshape(len(gename), 2)

    class6_rank_series = pd.Series(-feature_matrix[:, 6])
    class6_rank_ = class6_rank_series.rank()
    class6_rank = np.array(class6_rank_).reshape(len(class6_rank_), 1)
    gename_rank_6 = np.hstack((gename, class6_rank))
    gename_rank_6 = np.array(gename_rank_6).reshape(len(gename), 2)

    # # 对按从大到小进行排序
    # feature1 = abs(np.sort(-feature_matrix[:, 1]))
    # feature2 = abs(np.sort(-feature_matrix[:, 2]))
    # feature3 = abs(np.sort(-feature_matrix[:, 3]))
    # feature4 = abs(np.sort(-feature_matrix[:, 4]))
    # feature5 = abs(np.sort(-feature_matrix[:, 5]))
    # feature6 = abs(np.sort(-feature_matrix[:, 6]))
    #
    # # 作图，画出归一化后的所有基因的方差和均值变化的曲线图
    # plt.figure(1)
    # plt.subplot(231)
    # plt.plot(feature1[1:100])
    # plt.xlabel('Ranking')
    # plt.ylabel('Weights')
    # plt.title('The weights for class 1')
    # plt.grid(True)
    #
    # plt.subplot(232)
    # plt.plot(feature2[1:100])
    # plt.xlabel('Ranking')
    # plt.ylabel('Weights')
    # plt.title('The weights for class 2')
    # plt.grid(True)

    # plt.subplot(233)
    # plt.plot(feature3[1:100])
    # plt.xlabel('Ranking')
    # plt.ylabel('Weights')
    # plt.title('The weights for class 3')
    # plt.grid(True)
    #
    # plt.subplot(234)
    # plt.plot(feature4[1:100])
    # plt.xlabel('Ranking')
    # plt.ylabel('Weights')
    # plt.title('The weights for class 4')
    # plt.grid(True)
    #
    # plt.subplot(235)
    # plt.plot(feature5[1:100])
    # plt.xlabel('Ranking')
    # plt.ylabel('Weights')
    # plt.title('The weights for class 5')
    # plt.grid(True)
    #
    # plt.subplot(236)
    # plt.plot(feature6[1:100])
    # plt.xlabel('Ranking')
    # plt.ylabel('Weights')
    # plt.title('The weights for class 6')
    # plt.grid(True)
    #
    # plt.show()
    return gename_rank_1, gename_rank_2, gename_rank_3, gename_rank_4, gename_rank_5, gename_rank_6

def select_item(data, n):
    r, c = data.shape

    item = []
    for i in range(r):
        if data[i, 1] < n:
            item.append(data[i, 0])

    item = np.array(item)

    return item

def select_same_items(data1, data2):
    num = 0
    overlap_items = []
    for i in range(len(data1)):
        for j in range(len(data2)):
            if data1[i] == data2[j]:
                overlap_items.append(data2[j])
                num += 1
                break

    # print("重叠元素的数目为：", num)

    overlap_items = np.array(overlap_items)
    return overlap_items

def intersection(data1, data2, data3, data4, data5):
    item1 = select_item(data1, 1000)
    item2 = select_item(data2, 1000)
    item3 = select_item(data3, 1000)
    item4 = select_item(data4, 1000)
    item5 = select_item(data5, 1000)

    overlap_item1_2 = select_same_items(item1, item2)
    overlap_item1_2_3 = select_same_items(overlap_item1_2, item3)
    overlap_item1_2_3_4 = select_same_items(overlap_item1_2_3, item4)
    overlap_item1_2_3_4_5 = select_same_items(overlap_item1_2_3_4, item5)

    overlap_item1_2_3_4_5 = np.array(overlap_item1_2_3_4_5)
    print("重叠元素的数目为：", len(overlap_item1_2_3_4_5))

    return overlap_item1_2_3_4_5

def main():
    """
    主函数
    """
    # ========= Step 1. 读入数据 ===========
    # 关键基因及特征选择矩阵
    gename_2v80_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q80/1/gene_rankp.csv')
    feature_matrix_2v80_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q80/1/feature_matrix.csv')

    gename_2v92_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q92/1/gene_rankp.csv')
    feature_matrix_2v92_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q92/1/feature_matrix.csv')

    gename_2v111_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q111/1/gene_rankp.csv')
    feature_matrix_2v111_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q111/1/feature_matrix.csv')

    gename_2v140_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q140/1/gene_rankp.csv')
    feature_matrix_2v140_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q140/1/feature_matrix.csv')

    gename_2v175_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q175/1/gene_rankp.csv')
    feature_matrix_2v175_df = pd.read_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/q20_q175/1/feature_matrix.csv')

    # ========= Step 2. 得到每种比对中每个列别下的基因重要性的排名表 ===========
    # 排名越高，说明该类别下，基因的重要性越大
    gen_rank_2v80_1, gen_rank_2v80_2, gen_rank_2v80_3, gen_rank_2v80_4, gen_rank_2v80_5, gen_rank_2v80_6 = key_gene_rank(gename_2v80_df, 1, feature_matrix_2v80_df)
    gen_rank_2v92_1, gen_rank_2v92_2, gen_rank_2v92_3, gen_rank_2v92_4, gen_rank_2v92_5, gen_rank_2v92_6 = key_gene_rank(gename_2v92_df, 1, feature_matrix_2v92_df)
    gen_rank_2v111_1, gen_rank_2v111_2, gen_rank_2v111_3, gen_rank_2v111_4, gen_rank_2v111_5, gen_rank_2v111_6 = key_gene_rank(gename_2v111_df, 1, feature_matrix_2v111_df)
    gen_rank_2v140_1, gen_rank_2v140_2, gen_rank_2v140_3, gen_rank_2v140_4, gen_rank_2v140_5, gen_rank_2v140_6 = key_gene_rank(gename_2v140_df, 1, feature_matrix_2v140_df)
    gen_rank_2v175_1, gen_rank_2v175_2, gen_rank_2v175_3, gen_rank_2v175_4, gen_rank_2v175_5, gen_rank_2v175_6 = key_gene_rank(gename_2v175_df, 1, feature_matrix_2v175_df)

    # ======= Step 3. 保存文件 ============
    # 取每个类别中关键基因的交集 作为该类别的关键基因集合
    gen_rank_1 = intersection(gen_rank_2v80_1, gen_rank_2v92_1, gen_rank_2v111_1, gen_rank_2v140_1, gen_rank_2v175_1)
    gen_rank_2 = intersection(gen_rank_2v80_2, gen_rank_2v92_2, gen_rank_2v111_2, gen_rank_2v140_2, gen_rank_2v175_2)
    gen_rank_3 = intersection(gen_rank_2v80_3, gen_rank_2v92_3, gen_rank_2v111_3, gen_rank_2v140_3, gen_rank_2v175_3)
    gen_rank_4 = intersection(gen_rank_2v80_4, gen_rank_2v92_4, gen_rank_2v111_4, gen_rank_2v140_4, gen_rank_2v175_4)
    gen_rank_5 = intersection(gen_rank_2v80_5, gen_rank_2v92_5, gen_rank_2v111_5, gen_rank_2v140_5, gen_rank_2v175_5)
    gen_rank_6 = intersection(gen_rank_2v80_6, gen_rank_2v92_6, gen_rank_2v111_6, gen_rank_2v140_6, gen_rank_2v175_6)

    # ======= Step 4. 保存文件 ============
    # 将Numpy.array格式转化为pandas.dataframe格式
    gen_rank_1_df = pd.DataFrame(data = gen_rank_1)
    gen_rank_2_df = pd.DataFrame(data = gen_rank_2)
    gen_rank_3_df = pd.DataFrame(data=gen_rank_3)
    gen_rank_4_df = pd.DataFrame(data=gen_rank_4)
    gen_rank_5_df = pd.DataFrame(data=gen_rank_5)
    gen_rank_6_df = pd.DataFrame(data=gen_rank_6)

    # 对文件进行输出
    gen_rank_1_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/1/keygene_set_1.csv')
    gen_rank_2_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/1/keygene_set_2.csv')
    gen_rank_3_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/1/keygene_set_3.csv')
    gen_rank_4_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/1/keygene_set_4.csv')
    gen_rank_5_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/1/keygene_set_5.csv')
    gen_rank_6_df.to_csv('/Users/xuejiang/PycharmProjects/LPFS/project1/output/lpfs/keygene_set/1/keygene_set_6.csv')

if __name__ == '__main__':
    main()