# 特征分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from scipy import stats
warnings.filterwarnings('ignore')
def feature_distribution(columnOfDataFrame):
    """
    显示DataFrame 中一列的分布情况
    """
    print('描述性统计信息,你也可以自己使用下面的方法来做探索')
    print("""
编号     函数           描述
1       count()        非空观测数量
2       sum()          所有值之和
3       mean()         所有值的平均值
4       median()       所有值的中位数
5       mode()         值的模值
6       std()          值的标准偏差
7       min()          所有值中的最小值
8       max()          所有值中的最大值
9       abs()          绝对值
10      prod()         数组元素的乘积
11      cumsum()       累计总和
12      cumprod()      累计乘积
    """)
    print("---描述性信息统计---")
    print(columnOfDataFrame.describe(include='all'))
    # 散点分布
    plt.scatter(np.arange(len(columnOfDataFrame)), columnOfDataFrame, alpha=0.4, cmap='Reds')
    plt.grid()
    plt.show()
    # 散点分布

    sns.distplot(columnOfDataFrame)
    plt.show()

    """
    kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
    结果返回两个值：statistic → D值，pvalue → P值
    p值大于0.05，为正态分布
    H0:样本符合  
    H1:样本不符合 
    如何p>0.05接受H0 ,反之 
    """

    print("""
kstest方法：KS检验，
参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差

结果返回两个值：statistic → D值，pvalue → P值

p 值大于0.05，为正态分布
H0:样本符合  
H1:样本不符合 
如何 p>0.05 接受H0 ,反之 
    """)
    u = columnOfDataFrame.mean()
    std = columnOfDataFrame.std()
    result = stats.kstest(columnOfDataFrame, 'norm', (u, std))
    print(result)

    print("\n变量极差")
    print("Max(%f)-Min(%f) = %f"%(columnOfDataFrame.max(),columnOfDataFrame.min(),columnOfDataFrame.max()-columnOfDataFrame.min()))

    print("---频率分布情况---")
    columnOfDataFrame.hist(bins=10)
    plt.show()
    columnOfDataFrame.hist(bins=50)
    plt.show()

    print("---分组区间---")
    gcut = pd.cut(columnOfDataFrame, 10, right=False)
    gcut_count = gcut.value_counts(sort=False)
    # 在这里不排序
    columnOfDataFrame['分组区间'] = gcut.values
    # 给原表多加一列，写每列数据在的区间
    print(gcut.head(), '\n------')
    print(gcut_count)
    print(columnOfDataFrame.head())

    r_zj = pd.DataFrame(gcut_count)
    r_zj.rename(columns={gcut_count.name: '频数'}, inplace=True)  # 修改频数字段名
    r_zj['频率'] = r_zj / r_zj['频数'].sum()  # 计算频率
    r_zj['累计频率'] = r_zj['频率'].cumsum()  # 计算累计频率
    r_zj['频率%'] = r_zj['频率'].apply(lambda x: "%.2f%%" % (x * 100))  # 以百分比显示频率
    r_zj['累计频率%'] = r_zj['累计频率'].apply(lambda x: "%.2f%%" % (x * 100))  # 以百分比显示累计频率
    r_zj.style.bar(subset=['频率', '累计频率'], color='green', width=100)
    # pd.set_option("max_columns", None)  # Showing only two columns
    # pd.set_option("max_rows", None)
    print("---输出频*表---")
    print(r_zj)

    r_zj['频率'].plot(kind='bar', width=0.8, figsize=(12, 2), rot=0, color='k', grid=True, alpha=0.5)
    plt.title('参考总价分布频率直方图')

    x = len(r_zj)
    y = r_zj['频率']
    m = r_zj['频数']
    for i, j, k in zip(range(x), y, m):
        plt.text(i - 0.1, j + 0.01, '%i' % k, color='k')
    # 添加频数标签
    plt.show()

    plt.pie(r_zj['频数'],
            labels=r_zj.index,
            autopct='%.2f%%',
            shadow=True)
    plt.axis('equal')
    plt.show()
