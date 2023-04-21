# 特征分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')
from chinese_calendar import is_workday, is_holiday


def feature_distribution(datas, columnName):
    """
    显示DataFrame 中一列的分布情况
    :param datas DataFrame
    :param columnName 列名
    """

    columnOfDataFrame = datas[columnName]
    print("########################## Column " + columnName + '##########################')
    print('\n\n描述性统计信息,你也可以自己使用下面的方法来做探索')
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
    print("---" + columnName + " 描述性信息统计---")
    print(columnOfDataFrame.describe(include='all'))

    print('\n\n' + columnName + " 列中的唯一值和数量如下：\n")
    print(datas[columnName].value_counts())

    # 散点分布
    plt.title('Scatter')
    plt.scatter(np.arange(len(columnOfDataFrame)), columnOfDataFrame, alpha=0.4, cmap='Reds')
    plt.grid()
    plt.show()
    # 散点分布
    plt.title("Hist")
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

p 值大于0.05，为正态分布 H0:样本符合  H1:样本不符合 如何 p>0.05 接受H0 ,反之 
    """)
    try:
        u = columnOfDataFrame.mean()
        std = columnOfDataFrame.std()
        result = stats.kstest(columnOfDataFrame, 'norm', (u, std))
        print(result)

        print("变量极差", end='\t')
        print("Max(%f)-Min(%f) = %f" % (
        columnOfDataFrame.max(), columnOfDataFrame.min(), columnOfDataFrame.max() - columnOfDataFrame.min()))
    except Exception as e:
        print(e)
        pass
    print("---频率分布情况---")
    plt.title("Frequency Distribution Bin10")
    columnOfDataFrame.hist(bins=10)
    plt.show()
    plt.title("Frequency Distribution Bin50")
    columnOfDataFrame.hist(bins=50)
    plt.show()

    print("---分组区间---")
    gcut = pd.cut(columnOfDataFrame, 10, right=False)
    gcut_count = gcut.value_counts(sort=False)
    # 在这里不排序
    # columnOfDataFrame['分组区间'] = gcut.values
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
    plt.title('Distribution Hist')
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

    print("---箱线图---")
    print("""
简单直观的异常值检测方法:箱形图（箱线图）
箱形图中，从上到下依次有 6 个数据节点，分别是上界、上四分位、均值、中位数、下四分位、下界。而那些超过上界的值就会被标记为离群点，也就是异常数据。
    """)
    not_null = pd.to_numeric(columnOfDataFrame, errors='coerce')
    print(not_null)
    plt.boxplot(not_null)
    plt.show()

    print('\n\n')


def plot_scatter(datas, colX, colY, colHue):
    """
    显示散点图
    :param datas dataframe
    :param colX X 轴列，列名
    :param colY Y 轴列，列名
    :param colHue 数据显示列,字符串类型
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.lmplot(x=colX, y=colY,
               data=datas,
               hue=colHue,
               fit_reg=False)
    plt.xlabel(colX)
    plt.ylabel(colY)
    plt.title(colHue + 'Scatter Plot for ' + colX + " & " + colY)
    plt.show()


def count_unique(datas, cols):
    for col in cols:
        print('\n\n' + col + " 列中的唯一值和数量如下：\n")
        print(datas[col].value_counts())


# plot_bars(auto_prices, ['fuel_type'])
# plot_cols = ['make', 'body_style', 'num_of_cylinders']
# plot_bars(auto_prices, plot_cols)
def plot_bars(datas, cols):
    for col in cols:
        fig = plt.figure(figsize=(6, 6))  # 定义绘图区域
        ax = fig.gca()  # 定义轴axis
        counts = datas[col].value_counts()  # 找到每个唯一类别的计数
        counts.plot.bar(ax=ax, color='blue')  # 在计数数据框上使用 plot.bar 方法
        ax.set_title('Number of  by' + col)  # 给一个主标题
        ax.set_xlabel(col)  # 设置 x 轴的文本
        ax.set_ylabel('Numbers')  # 为 y 轴设置文本
        plt.show()


# num_cols = ['curb_weight', 'engine_size', 'city_mpg', 'price']
# plot_histogram(auto_prices, num_cols)
def plot_histogram(datas, cols, bins=10):
    for col in cols:
        fig = plt.figure(figsize=(6, 6))  # define plot area
        ax = fig.gca()  # define axis
        datas[col].plot.hist(ax=ax, bins=bins)  # Use the plot.hist method on subset of the data frame
        ax.set_title('Histogram of ' + col)  # Give the plot a main title
        ax.set_xlabel(col)  # Set text for the x axis
        ax.set_ylabel('Numbers')  # Set text for y axis
        plt.show()


# plot_density_hist(auto_prices, num_cols, bins = 20, hist = True)
def plot_density_hist(datas, cols, bins=10, hist=False):
    for col in cols:
        sns.set_style("whitegrid")
        sns.distplot(datas[col], bins=bins, rug=True, hist=hist)
        plt.title('Histogram of ' + col)  # Give the plot a main title
        plt.xlabel(col)  # Set text for the x axis
        plt.ylabel('Numbers')  # Set text for y axis
        plt.show()


# plot_scatter(auto_prices, ['horsepower'], 'engine_size')
# num_cols = ['curb_weight', 'engine_size', 'horsepower', 'city_mpg']
# plot_scatter(auto_prices, num_cols)
def plot_scatter(datas, cols, col_y='price'):
    for col in cols:
        fig = plt.figure(figsize=(7, 6))  # define plot area
        ax = fig.gca()  # define axis
        datas.plot.scatter(x=col, y=col_y, ax=ax)
        ax.set_title('Scatter plot of ' + col_y + ' vs. ' + col)  # Give the plot a main title
        ax.set_xlabel(col)  # Set text for the x axis
        ax.set_ylabel(col_y)  # Set text for y axis
        plt.show()


# plot_desity_2d(auto_prices, num_cols)
# plot_desity_2d(auto_prices, num_cols, kind = 'hex')
def plot_desity_2d(datas, cols, col_y='price', kind='kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=datas, kind=kind)
        plt.xlabel(col)  # Set text for the x axis
        plt.ylabel(col_y)  # Set text for y axis
        plt.show()


# cat_cols = ['fuel_type', 'aspiration', 'num_of_doors', 'body_style',
#             'drive_wheels', 'engine_location', 'engine_type', 'num_of_cylinders']
# plot_box(auto_prices, cat_cols)
def plot_box(datas, cols, col_y='price'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col, col_y, data=datas)
        plt.xlabel(col)  # Set text for the x axis
        plt.ylabel(col_y)  # Set text for y axis
        plt.show()


# plot_violin(auto_prices, cat_cols)
def plot_violin(datas, cols, col_y='price'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col, col_y, data=datas)
        plt.xlabel(col)  # Set text for the x axis
        plt.ylabel(col_y)  # Set text for y axis
        plt.show()


# num_cols = ['curb_weight', 'engine_size', 'horsepower', 'city_mpg']
# plot_scatter_shape(auto_prices, num_cols)
def plot_scatter_shape(datas, cols, shape_col='fuel_type', col_y='price', alpha=0.2):
    shapes = ['+', 'o', 's', 'x', '^']  # pick distinctive shapes
    unique_cats = datas[shape_col].unique()
    for col in cols:  # loop over the columns to plot
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats):  # loop over the unique categories
            temp = datas[datas[shape_col] == cat]
            sns.regplot(col, col_y, data=temp, marker=shapes[i], label=cat,
                        scatter_kws={"alpha": alpha}, fit_reg=False, color='blue')
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col)  # Give the plot a main title
        plt.xlabel(col)  # Set text for the x axis
        plt.ylabel(col_y)  # Set text for y axis
        plt.legend()
        plt.show()


# num_cols = ['engine_size', 'horsepower', 'city_mpg']
# plot_scatter_size(auto_prices, num_cols)
def plot_scatter_size(datas, cols, shape_col='fuel_type', size_col='curb_weight',
                      size_mul=0.000025, col_y='price', alpha=0.2):
    shapes = ['+', 'o', 's', 'x', '^']  # pick distinctive shapes
    unique_cats = datas[shape_col].unique()
    for col in cols:  # loop over the columns to plot
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats):  # loop over the unique categories
            temp = datas[datas[shape_col] == cat]
            sns.regplot(col, col_y, data=temp, marker=shapes[i], label=cat,
                        scatter_kws={"alpha": alpha, "s": size_mul * temp[size_col] ** 2},
                        fit_reg=False, color='blue')
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col)  # Give the plot a main title
        plt.xlabel(col)  # Set text for the x axis
        plt.ylabel(col_y)  # Set text for y axis
        plt.legend()
        plt.show()


# num_cols = ['engine_size', 'horsepower', 'city_mpg']
# plot_scatter_shape_size_col(auto_prices, num_cols)
def plot_scatter_shape_size_col(datas, cols, shape_col='fuel_type', size_col='curb_weight',
                                size_mul=0.000025, color_col='aspiration', col_y='price', alpha=0.2):
    shapes = ['+', 'o', 's', 'x', '^']  # pick distinctive shapes
    colors = ['green', 'blue', 'orange', 'magenta', 'gray']  # specify distinctive colors
    unique_cats = datas[shape_col].unique()
    unique_colors = datas[color_col].unique()
    for col in cols:  # loop over the columns to plot
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats):  # loop over the unique categories
            for j, color in enumerate(unique_colors):
                temp = datas[(datas[shape_col] == cat) & (datas[color_col] == color)]
                sns.regplot(col, col_y, data=temp, marker=shapes[i],
                            scatter_kws={"alpha": alpha, "s": size_mul * temp[size_col] ** 2},
                            label=(cat + ' and ' + color), fit_reg=False, color=colors[j])
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col)  # Give the plot a main title
        plt.xlabel(col)  # Set text for the x axis
        plt.ylabel(col_y)  # Set text for y axis
        plt.legend()
        plt.show()


# plot_violin_hue(auto_prices, cat_cols)
def plot_violin_hue(datas, cols, col_y='price', hue_col='aspiration'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col, col_y, data=datas, hue=hue_col, split=True)
        plt.xlabel(col)  # Set text for the x axis
        plt.ylabel(col_y)  # Set text for y axis
        plt.show()


# num_cols = ["curb_weight", "engine_size", "horsepower", "city_mpg", "price", "fuel_type"]
def plot_scatter_pairplot(datas, num_cols):
    sns.pairplot(datas[num_cols],
                 hue='fuel_type',
                 palette="Set2",
                 diag_kind="kde",
                 size=2).map_upper(sns.kdeplot, cmap="Blues_d")


## Define columns for making a conditioned histogram
# plot_cols2 = ["length",
#                "curb_weight",
#                "engine_size",
#                "city_mpg",
#                "price"]
#
# cond_hists(auto_prices, plot_cols2, 'drive_wheels')
## Function to plot conditioned histograms
def cond_hists(df, plot_cols, grid_col):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ## Loop over the list of columns
    for col in plot_cols:
        grid1 = sns.FacetGrid(df, col=grid_col)
        grid1.map(plt.hist, col, alpha=.7)
    return grid_col


def sigma3(x):
    '''
    MBA智库对3σ原则的描述：

    σ代表标准差,μ代表均值

    样本数据服从正态分布的情况下

    数值分布在（μ-σ,μ+σ)中的概率为0.6826

    数值分布在（μ-2σ,μ+2σ)中的概率为0.9544

    数值分布在（μ-3σ,μ+3σ)中的概率为0.9974

    可以认为，Y 的取值几乎全部集中在（μ-3σ,μ+3σ)区间内，超出这个范围的可能性仅占不到0.3%。

    https://www.guofei.site/2017/10/19/cleandata.html
    '''
    x = pd.Series(x)
    mean_ = x.mean()
    std_ = x.std()
    rules = (mean_ - 3 * std_ > x) | (mean_ + 3 * std_ < x)
    indx = x[rules].index
    # 获取异常值
    # out = x[indx]
    return indx


def eda_profile(data):
    """
    ydata_profile
    """
    from ydata_profiling import ProfileReport
    profile = ProfileReport(data, title="Profiling Report")
    profile.to_file("data_analysis.html")


def eda_pgw(data):
    import pygwalker as pyg
    gwalker = pyg.walk(data)


## 日期相关操作
def month_stage(x):
    if x in range(1, 11):
        return 0  # 上旬
    elif x in range(11, 21):
        return 1  # 中旬
    else:
        return 2  # 下旬


# time
def time_feature(data, col):
    """
    对日期类型的列进行处理，分割出来更多的字段、特征、列
    """
    data['order_date'] = pd.to_datetime(data[col])
    data['dayofmonth'] = data[col].dt.day
    data['dayofweek'] = data[col].dt.dayofweek
    data['month'] = data[col].dt.month
    data['year'] = data[col].dt.year
    data['is_month_start'] = (data[col].dt.is_month_start).astype(int)
    data['is_month_end'] = (data[col].dt.is_month_end).astype(int)
    data['is_workday'] = (data[col].apply(lambda x: is_workday(x))).astype(int)
    data['is_holiday'] = (data[col].apply(lambda x: is_holiday(x))).astype(int)
    data['in_quarter'] = data[col].dt.quarter
    data['in_month_stage'] = data['dayofmonth'].apply(month_stage)
    return data


##

def segments_bins_labels(datas, col, bins, labels):
    """
    按段为原来的 DataFrame 增加新的字段
    :param datas DataFrame
    :param col 要处理的列
    :param bins 分割规则
    :param lables 分割后给的描述信息
    """

    segments = pd.cut(datas[col], bins, labels)
    datas['segements_' + col] = segments
    return datas
