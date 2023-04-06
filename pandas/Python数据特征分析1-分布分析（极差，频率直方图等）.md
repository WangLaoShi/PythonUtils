## 数据特征分析分为以下部分：

1. 分布分析
2. 对比分析
3. 统计分析
4. 帕累托分析
5. 正态性检验
6. 相关性分析

## 数据：

![pedxoR](https://oss.images.shujudaka.com/uPic/pedxoR.jpg)

## 分布分析

分布分析 --> 研究数据的分布特征和分布类型，分定量数据、定性数据

主要是：极差、频率分布情况、分组组距及组数

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```

```python
#作散点图：横纵轴放经纬度，单价显示大小，总价显示颜色
data = pd.read_csv('./datas/second_hand_ house.csv')
data.head()
# matplotlib.pyplot.scatter(x, y, s=20, c='b', marker='o', 
# 	cmap=None, norm=None, vmin=None, vmax=None, alpha=None, 
# 	linewidths=None, verts=None, hold=None, **kwargs)
# x，y：表示的是shape大小为(n,)的数组，也就是我们即将绘制散点图的数据点，输入数据。
# s：表示的是大小，是一个标量或者是一个shape大小为(n,)的数组，可选，默认20。
# c：表示的是色彩或颜色序列，可选，默认蓝色’b’。但是c不应该是一个单一的RGB数字，也不应该是一个RGBA的序列，
# 因为不便区分。c可以是一个RGB或RGBA二维行数组。
# b---blue c---cyan g---green k---black
# m---magenta r---red w---white  y---byellow
# marker：MarkerStyle，表示的是标记的样式，可选，默认’o’。
# cmap：Colormap，标量或者是一个colormap的名字，cmap仅仅当c是一个浮点数数组的时候才使用。如果没有申明就是image.cmap，可选，默认None。
# norm：Normalize，数据亮度在0-1之间，也是只有c是一个浮点数的数组的时候才使用。如果没有申明，就是默认None。
# vmin，vmax：标量，当norm存在的时候忽略。用来进行亮度数据的归一化，可选，默认None。
# alpha：标量，0-1之间，可选，默认None。
# linewidths：也就是标记点的长度，默认None。


plt.scatter(data['经度'],data['纬度'],
            s = data['房屋单价']/500,
            c = data['参考总价'],
            alpha=0.4,
            cmap = 'Reds')
plt.grid()
print(data.dtypes) #显示各列类型
print('------\n数据长度%i条'%len(data)) #输出数据长度

```