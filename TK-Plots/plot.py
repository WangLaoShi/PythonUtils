"""
    绘图文件，负责函数曲线的绘制
"""
import numpy as np


def plot_main(string, plt):
    """
        负责函数曲线的绘制
    :param string: 数学表达式
    :param plt: 画布的对象
    :return: 无
    """
    list_expr = []
    list_expr = string.split(",")
    string1 = []
    for sub_expr in list_expr:
        string1.append(sub_expr)
    x = np.linspace(-10, 10, 100)
    y = []
    num = string.count('x')
    for i in x:
        t = (i, ) * num
        string = string.replace("x", "(%f)")
        i = eval(string % t)
        y.append(i)
    plt.plot(x, y)
    plt.grid(True)
    plt.legend(labels=string1)
