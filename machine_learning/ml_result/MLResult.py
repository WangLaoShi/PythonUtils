from sklearn.model_selection import learning_curve  # 导入学习曲线类
from machine_learning.model_dump_load import dump_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from machine_learning.ml_classifier.classification import *
from machine_learning.ml_regression.regression import *

# classification_model_methods = ['KNN', 'SVM', "LR", "CART", "RF", "GBDT", 'GNB', 'BNB' , 'AdaBoost']
# classification_model_methods = ["CART", "RF", "GBDT", 'AdaBoost']

classification_model_methods = ["CART",
                                "CARTTuning",
                                "RF",
                                "RFTuning",
                                "GBDT",
                                "GBDTTuning",
                                'AdaBoost',
                                'AdaBoostTuning']

regression_model_methods = ["Linear",
                            "Ridge",
                            "Lasso",
                            "Elastic",
                            "DecisionTree",
                            # "SVR",
                            'KNN',
                            'Bagging',
                            'RF',
                            'ExtraTree',
                            'ADA',
                            'GB'
                            ]
def classifier_selection(method, X_train, y_train, X_test):
    """
    要执行的函数的名字
    :param X_test:
    :param y_train:
    :param X_train:
    :param method:
    :return:
    """
    category = ClassifierCollection()
    if hasattr(category, method):  # 判断在commons模块中是否存在inp这个字符串
        target_func = getattr(category, method)  # 获取inp的引用
        return target_func(X_train, y_train, X_test)  # 执行

def regressor_selection(method, X_train, y_train, X_test):
    """
    要执行的函数的名字
    :param X_test:
    :param y_train:
    :param X_train:
    :param method:
    :return:
    """
    category = RegressorCollection()
    if hasattr(category, method):  # 判断在commons模块中是否存在inp这个字符串
        target_func = getattr(category, method)  # 获取inp的引用
        return target_func(X_train, y_train, X_test)  # 执行

def outputScoreInConsole(accScoreDict,
                         recallScoreDict,
                         f1ScoreDict,
                         precisionScoreDict):
    print("准确率 ",end='')
    print(accScoreDict)

    print("召回率 ",end='')
    print(recallScoreDict)

    print("F1    ",end='')
    print(f1ScoreDict)

    print("精准率 ",end='')
    print(precisionScoreDict)


def cls_ml_scores(X_train, y_train, X_test, y_test):
    """
    使用了反射，可能较难理解，为不同的自定义分类方法，不同的数据填充方法，不同的分类方法打分
    :param task 任务类型，可以从 cls 分类，rgs 回归，2 个当中选择，下面会有判断。
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """

    # https://blog.csdn.net/sinat_26917383/article/details/75199996
    accScoreDict = {}
    recallScoreDict = {}
    f1ScoreDict = {}
    precisionScoreDict = {}

    methods = list()

    methods = classification_model_methods

    # 使用反射对 8 种分类方法进行运行，并计算分数
    for method in methods:
        print("当前运行参数------->")
        print("算法 ", method, X_train.shape, X_test.shape)
        print("当前运行参数------->")

        # 返回预测值和模型
        _predicted, model = classifier_selection(method, X_train, y_train, X_test)

        print(" Classfication method----->",method)
        # 将模型保存起来
        dump_model(model, method)

        # 各种分数
        # https://blog.csdn.net/lyb3b3b/article/details/84819931
        accScoreDict[method] = accuracy_score(y_test, _predicted)  # 准确率,正确率（accuracy）
        # ValueError: Target is multiclass but average='binary'.
        # Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
        recallScoreDict[method] = recall_score(y_test, _predicted, average='micro')  # 召回率
        f1ScoreDict[method] = f1_score(y_test, _predicted, average='micro')  # F1
        precisionScoreDict[method] = precision_score(y_test, _predicted, average='micro')  # 精准率
        # 各种分数

        print("当前运行算法", method)
        outputScoreInConsole(accScoreDict, recallScoreDict, f1ScoreDict, precisionScoreDict)

    # 各种分数
    acc_scores = list()
    recall_scores = list()
    f1_scores = list()
    precision_scores = list()
    for method in methods:
        acc_scores.append(accScoreDict[method])
        recall_scores.append(recallScoreDict[method])
        f1_scores.append(f1ScoreDict[method])
        precision_scores.append(precisionScoreDict[method])
    # 各种分数

    scoresDF = pd.DataFrame().from_dict({
        'method': methods,
        'acc_score': acc_scores,
        'recall_score': recall_scores,
        'f1_score': f1_scores,
        'precision_score': precision_scores
    })

    scoresDF.sort_values('acc_score', inplace=True)

    # 保存各种方法下的分类准确率，为之后的集成学习对比做准备。
    scoresDF.to_csv(
        './results-storage/classification_results/classfication.csv')

    # https://jakevdp.github.io/PythonDataScienceHandbook/04.01-simple-line-plots.html
    # Draw plot
    import matplotlib.patches as patches
    # import seaborn as sns
    #
    # plots = sns.barplot(x="method", y="score", data=scoresDF)
    #
    # # Iterating over the bars one-by-one
    # for bar in plots.patches:
    #     # Using Matplotlib's annotate function and
    #     # passing the coordinates where the annotation shall be done
    #     plots.annotate(format(bar.get_height(), '.2f'),
    #                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
    #                    ha='center', va='center',
    #                    size=15, xytext=(0, 5),
    #                    textcoords='offset points')
    plt.ylim(0, 1.2);
    plt.plot(scoresDF['method'], scoresDF['acc_score'], color='blue', label='acc_score')
    plt.plot(scoresDF['method'], scoresDF['recall_score'], color='g', label='recall_score')
    plt.plot(scoresDF['method'], scoresDF['f1_score'], color='#FFDD44', label='f1_score')
    plt.plot(scoresDF['method'], scoresDF['precision_score'], color='0.75', label='precision_score')
    plt.xlabel("Method")
    plt.ylabel("Score")

    # Title, Label, Ticks and Ylim
    plt.title('Bar Chart for cls' ,fontdict={'size': 22})

    # Add patches to color the X axis labels
    # p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1, facecolor='green', transform=fig.transFigure)
    # p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1, facecolor='red', transform=fig.transFigure)
    # fig.add_artist(p1)
    # fig.add_artist(p2)
    fileName = './results-storage/charts/score-barcharts/BarChartfor-cls' + ".png"
    plt.savefig(fileName)
    plt.show()

    return fileName


def plot_learn_curve(task,X_train, y_train, X_test):
    """
    使用了反射，可能较难理解，为不同的自定义分类方法，不同的数据填充方法，不同的分类方法打分
    :param X_train:
    :param y_train:
    :return:
    """

    scoreDict = {}
    methods = list()
    if task =='cls':
        methods = classification_model_methods
    elif task =='rgs':
        methods = regression_model_methods

    # 使用反射对 8 种分类方法进行运行，并计算分数
    for method in methods:
        print("*" * 10, '方法', methods,
               "--模型方法", method, " learn Curve", "*" * 10)
        # 返回预测值和模型
        if task == 'cls':
            _predicted, model = classifier_selection(method, X_train, y_train, X_test)
        elif task == 'rgs':
            _predicted, model = regressor_selection(method, X_train, y_train, X_test)

        plot_lc(X_train, y_train, model, method)


def plot_lc(x, y, model, class_model):
    """

    :param x:
    :param y:
    :param model:
    :param class_model: 模型
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))  # 设置画布和子图

    train_sizes, train_scores, test_scores = learning_curve(model,
                                                            x,
                                                            y,
                                                            cv=20,
                                                            n_jobs=4)
    # 设置分类器为随机森林，x，y，5折交叉验证，cpu同时运算为4个
    ax.set_ylim((0.5, 1.1))  # 设置子图的纵坐标的范围为（0.7~1.1）
    ax.set_xlabel("模型" + class_model)  # 设置子图的x轴名称
    ax.set_ylabel("score")
    ax.grid()  # 画出网图
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='train score')
    # 画训练集数据分数，横坐标为用作训练的样本数，纵坐标为不同折下的训练分数的均值
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='test score')
    ax.legend(loc='best')  # 设置图例
    plt.savefig(
        "./results-storage/charts/learn_curve/" + class_model + "_learn_curve.png")
    plt.show()

def rgs_ml_scores(X_train, y_train, X_test, y_test):
    methods = list()

    methods = regression_model_methods

    # 使用反射对 8 种分类方法进行运行，并计算分数
    for method in methods:
        print("当前运行参数------->")
        print("算法 ", method, X_train.shape, X_test.shape)
        print("当前运行参数------->")

        # 返回预测值和模型
        _predicted, model = regressor_selection(method, X_train, y_train, X_test)

        print(" Regression method----->", method)
        # 将模型保存起来
        dump_model(model, method)

        lin_regplot(X_train,y_train,X_test,y_test,model)

def lin_regplot(X_train, y_train,X_test,y_test, model):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
    # 预测值与偏差的关系
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
    plt.xlim([-10, 50])
    plt.tight_layout()
    # plt.savefig('./figures/slr_residuals.png', dpi=300)
    plt.show()

    # ![U5vnAA](https://oss.images.shujudaka.com/uPic/U5vnAA.png)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error  # 均方误差回归损失
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
    print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))
    # Output:
    # MSE train: 19.958, test: 27.196
    # R^2 train: 0.765, test: 0.673