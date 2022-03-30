"""
    主程序文件，负责程序的启动与结束和窗体的大致设置。
"""

import tkinter as tk
import widget


def win_w_h(root):
    """
        控制窗口的大小和出现的位置
    :param root:
    :return: 窗口的大小和出现的位置
    """
    # 设置标题：
    win.title("数学函数绘图")

    # 绘图区标签
    label_plot = tk.Label(root, text="绘     图       区",
                          font=("微软雅黑", 10), fg="blue")
    label_plot.place(relx=0.26, rely=0)

    label_func = tk.Label(root, text="功     能       区",
                          font=("微软雅黑", 10), fg="blue")
    label_func.place(relx=0.75, rely=0)
    # 获取屏幕的大小;
    screen_height = root.winfo_screenheight()
    screen_width = root.winfo_screenwidth()
    # 窗体的大小
    win_width = 0.8 * screen_width
    win_height = 0.8 * screen_height
    # 窗体出现的位置：控制的是左上角的坐标
    show_width = (screen_width - win_width) / 2
    show_height = (screen_height - win_height) / 2

    # 返回窗体 坐标
    return win_width, win_height, show_width, show_height


win = tk.Tk()
# 大小 位置
win.geometry("%dx%d+%d+%d" % (win_w_h(win)))


# 创建一个容器, 没有画布时的背景
frame1 = tk.Frame(win, bg="#c0c0c0")
frame1.place(relx=0.00, rely=0.05, relwidth=0.62, relheight=0.89)


# 控件区
frame2 = tk.Frame(win, bg="#808080")
frame2.place(relx=0.62, rely=0.05, relwidth=0.38, relheight=0.89)

# 调用控件模块
widget.widget_main(win, frame2)
win.mainloop()
