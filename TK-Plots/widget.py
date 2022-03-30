"""
    控件文件，负责程序控件的创建与布局
"""
import tkinter as tk
# 对话框所需的库
import tkinter.messagebox as mb
# 画布文件
import figure
# 绘图文件
import plot


def widget_main(win, root):
    """
        负责程序控件的创建与布局
    :param win: 主窗体的对象。
    :param root: 绘图区的容器对象。
    :return: 无
    """
    # 控件区的容器对象
    frame1 = None


# ===========功能区============================
    # 绘图的功能函数
    def plot_f():
        string = entry.get()
        # 判断输入框是否为空
        if string == "":
            mb.showerror("提示", "没有输入值，请重新输入：")
        else:
            # 判断是否已经创建画布
            if frame1==None:
                mb.showerror("提示", "没有创建画布，不能画图，请先创建画布")
            else:
                axs = figure.plot_fun(frame1)
                plot.plot_main(string, axs)

    # 清除的功能函数
    def clear():
        nonlocal frame1
        entry.delete(0, "end")
        if frame1==None:
            mb.showerror("提示", "已经没有画布，无法清除画布")
        else:
            frame1.destroy()
            frame1 = None

    # 创建画布的功能函数
    def create():
        nonlocal frame1
        if frame1 != None:
            mb.showerror("提示", "画布已经存在，请不要重复创建画布")
        else:
            frame1 = tk.LabelFrame(win, bg="#80ff80", text="画-----布", labelanchor="n", fg="green")
            frame1.place(relx=0.00, rely=0.05, relwidth=0.62, relheight=0.95)


# =============控件区======================
    #  标签控件
    label = tk.Label(root,
                     text="请输入含x的数学公式：",
                     font=("微软雅黑", 18),
                     fg="blue")
    label.place(relx=0.2, rely=0.1)

    # 输入框
    entry = tk.Entry(root, font=("华文楷体", 15))
    entry.place(relx=0.1, rely=0.2, relwidth=0.8)

    # 创建画布区
    btn_draw = tk.Button(root,
                         text="创建",
                         cursor="hand2",
                         width=10,
                         bg="orange",
                         relief="raised",
                         command=create
                         )
    btn_draw.place(relx=0.1, rely=0.3)

    # 绘图按钮
    btn_draw = tk.Button(root,
                         text="绘图",
                         cursor="hand2",
                         width=10,
                         bg="green",
                         relief="raised",
                         command=plot_f
                         )
    btn_draw.place(relx=0.4, rely=0.3)

    # 清除按钮
    btn_clear = tk.Button(root,
                          text="清除",
                          cursor="hand2",
                          width=10,
                          bg="yellow",
                          relief="raised",
                          command=clear
                          )
    btn_clear.place(relx=0.7, rely=0.3)
