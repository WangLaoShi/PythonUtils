"""
    画布文件，实现绘图区域的显示，并返回画布的对象。
"""
import tkinter as tk

# 创建画布需要的库
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# 创建工具栏需要的库
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
# 快捷键需要的库
from matplotlib.backend_bases import key_press_handler
# 导入画图常用的库
from matplotlib.figure import Figure

def plot_fun(root):
    """
        该函数实现的是内嵌画布,不负责画图,返回画布对象。
    :param root:父亲控件对象, 一般是容器控件或者窗体
    :return: 画布对象
    """
    # 画布的大小和分别率
    fig = Figure(dpi=100)
    axs = fig.add_subplot(111)

    # 创建画布
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    # 显示画布
    canvas.get_tk_widget().pack()

    # 创建工具条
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    # 显示工具条
    canvas.get_tk_widget().pack()

    # 调用快捷键
    def on_key_press(event):
        key_press_handler(event, canvas, toolbar)

    canvas.mpl_connect("key_press_event", on_key_press)

    # 返回画布的对象
    return axs
