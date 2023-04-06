# 特征分析

def feature_distribution(columnOfDataFrame):
    """
    显示DataFrame 中一列的分布情况
    """
    print(columnOfDataFrame.describe())