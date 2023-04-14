import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from model_dump_load import dump_model
from sklearn.model_selection import learning_curve  # 导入学习曲线类
import numpy as np

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

# 这个类使用了反射，比较难。
class ClassifierCollection:

    ## KNN
    @staticmethod
    def KNN(X, y, XX):  # X,y 分别为训练数据集的数据和标签，XX为测试数据
        """
        KNN 分类
        :param X:
        :param y:
        :param XX:
        :return:
        """
        model = KNeighborsClassifier(n_neighbors=10)  # 默认为5
        model.fit(X, y)
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def SVM(X, y, XX):
        """
        SVM 分类
        :param X:
        :param y:
        :param XX:
        :return:
        """
        model = SVC(C=5.0)
        model.fit(X, y)
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def LR(X, y, XX):
        """
        LogisticRegression
        :param X:
        :param y:
        :param XX:
        :return:
        """
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
        model.fit(X, y)
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def CART(X, y, XX):
        """
        决策树（CART）
        :param X:
        :param y:
        :param XX:
        :return:
        """
        model = DecisionTreeClassifier()
        model.fit(X, y)
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def CARTTuning(X, y, XX):
        """
        决策树（CART）
        :param X:
        :param y:
        :param XX:
        :return:
        """
        model = DecisionTreeClassifier(random_state=42, max_depth=10, max_leaf_nodes=120)
        model.fit(X, y)
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def RF(X, y, XX):
        """
        随机森林
        :param X:
        :param y:
        :param XX:
        :return:
        """
        model = RandomForestClassifier()

        model.fit(X, y)
        # print("RandomForestClassifier 使用的分类器")
        # print(model.estimators_)
        # print("RandomForestClassifier 使用的分类器")
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def RFTuning(X, y, XX):
        """
        随机森林
        :param X:
        :param y:
        :param XX:
        :return:
        """
        model = RandomForestClassifier(random_state=42, max_depth=10, max_leaf_nodes=120)

        model.fit(X, y)
        # print("RandomForestClassifier 使用的分类器")
        # print(model.estimators_)
        # print("RandomForestClassifier 使用的分类器")
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def GBDT(X, y, XX):
        """
        (Gradient Boosting Decision Tree)
        :param X:
        :param y:
        :param XX:
        :return:
        """
        model = GradientBoostingClassifier()
        # https://blog.csdn.net/tuanzide5233/article/details/104234246

        model.fit(X, y)
        # print("GradientBoostingClassifier 使用的分类器")
        # print(model.estimators_)
        # print("GradientBoostingClassifier 使用的分类器")
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def GBDTTuning(X, y, XX):
        """
        (Gradient Boosting Decision Tree)
        :param X:
        :param y:
        :param XX:
        :return:
        """
        model = GradientBoostingClassifier(random_state=42,
                                           max_depth=10,
                                           max_leaf_nodes=120,
                                           n_estimators=100)
        # https://blog.csdn.net/tuanzide5233/article/details/104234246

        model.fit(X, y)
        # print("GradientBoostingClassifier 使用的分类器")
        # print(model.estimators_)
        # print("GradientBoostingClassifier 使用的分类器")
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def GNB(X, y, XX):
        """
        基于高斯分布求概率
        :param X:
        :param y:
        :param XX:
        :return:
        """
        model = GaussianNB()
        model.fit(X, y)
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def MNB(X, y, XX):
        """
        基于多项式分布求概率
        :param X:
        :param y:
        :param XX:
        :return:
        """
        model = MultinomialNB()
        model.fit(X, y)
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def BNB(X, y, XX):
        """
        基于伯努利分布求概率
        :param X:
        :param y:
        :param XX:
        :return:
        """
        model = BernoulliNB()
        model.fit(X, y)
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def AdaBoost(X, y, XX):
        """
        AdaBoost分类器
        :param X:
        :param y:
        :param XX:
        :return:
        """
        # 要把随机森林、GBDT、AdaBoost的弱分类器设置成CART 2023年02月17日 [DONE]
        # ![soRpwZ](https://oss.images.shujudaka.com/uPic/soRpwZ.png)
        model = AdaBoostClassifier(n_estimators=10, random_state=0)

        # AdaBoostClassifier默认使用CART分类树DecisionTreeClassifier，
        # 而AdaBoostRegressor默认使用CART回归树DecisionTreeRegressor。
        model.fit(X, y)
        # print("AdaBoostClassifier 使用的分类器")
        # print(model.estimators_)
        # print("AdaBoostClassifier 使用的分类器")
        predicted = model.predict(XX)
        return predicted, model

    @staticmethod
    def AdaBoostTuning(X, y, XX):
        """
        AdaBoost分类器
        :param X:
        :param y:
        :param XX:
        :return:
        """
        # 要把随机森林、GBDT、AdaBoost的弱分类器设置成CART 2023年02月17日 [DONE]
        # ![soRpwZ](https://oss.images.shujudaka.com/uPic/soRpwZ.png)
        model = AdaBoostClassifier(random_state=42,
                                   base_estimator=DecisionTreeClassifier(max_depth=10,max_leaf_nodes=120))

        # AdaBoostClassifier默认使用CART分类树DecisionTreeClassifier，
        # 而AdaBoostRegressor默认使用CART回归树DecisionTreeRegressor。
        model.fit(X, y)
        # print("AdaBoostClassifier 使用的分类器")
        # print(model.estimators_)
        # print("AdaBoostClassifier 使用的分类器")
        predicted = model.predict(XX)
        return predicted, model


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


def classifier_scores(X_train, y_train, X_test, y_test,
                      classificationMethod, missingValueFillMethodName,
                      columnSplitMethod='Default'):
    """
    使用了反射，可能较难理解，为不同的自定义分类方法，不同的数据填充方法，不同的分类方法打分
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param classificationMethod:['M1','M2',"M3',"M4"] 自定义分类方法
    :param missingValueFillMethodName:['knn', 'rf', 'interpolate'] 填充方法
    :param columnSplitMethod: 数据分割方法
    :return:
    """

    # https://blog.csdn.net/sinat_26917383/article/details/75199996
    accScoreDict = {}
    recallScoreDict = {}
    f1ScoreDict = {}
    precisionScoreDict = {}

    # 使用反射对 8 种分类方法进行运行，并计算分数
    for class_model in classification_model_methods:
        print("当前运行参数------->")
        print("分类算法 ", class_model, X_train.shape, X_test.shape)
        print("当前运行参数------->")

        # 返回预测值和模型
        _predicted, model = classifier_selection(class_model, X_train, y_train, X_test)
        print("ClassModel----->",class_model)
        # 将模型保存起来
        dump_model(model, class_model)

        # 各种分数
        # https://blog.csdn.net/lyb3b3b/article/details/84819931
        accScoreDict[class_model] = accuracy_score(y_test, _predicted)  # 准确率
        # ValueError: Target is multiclass but average='binary'.
        # Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
        recallScoreDict[class_model] = recall_score(y_test, _predicted, average='micro')  # 召回率
        f1ScoreDict[class_model] = f1_score(y_test, _predicted, average='micro')  # F1
        precisionScoreDict[class_model] = precision_score(y_test, _predicted, average='micro')  # 精准率
        # 各种分数

        print("当前运行算法", class_model)
        outputScoreInConsole(accScoreDict, recallScoreDict, f1ScoreDict, precisionScoreDict)

    # 各种分数
    acc_scores = list()
    recall_scores = list()
    f1_scores = list()
    precision_scores = list()
    for method in classification_model_methods:
        acc_scores.append(accScoreDict[method])
        recall_scores.append(recallScoreDict[method])
        f1_scores.append(f1ScoreDict[method])
        precision_scores.append(precisionScoreDict[method])
    # 各种分数

    scoresDF = pd.DataFrame().from_dict({
        'method': classification_model_methods,
        'acc_score': acc_scores,
        'recall_score': recall_scores,
        'f1_score': f1_scores,
        'precision_score': precision_scores
    })

    scoresDF.sort_values('acc_score', inplace=True)

    # 保存各种方法下的分类准确率，为之后的集成学习对比做准备。
    scoresDF.to_csv(
        './results-storage/classification_results/' + columnSplitMethod + '-' + missingValueFillMethodName + "-" + classificationMethod + ".csv")

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
    plt.title('Bar Chart for ' + columnSplitMethod + '-' + missingValueFillMethodName + "-" + classificationMethod,
              fontdict={'size': 22})

    # Add patches to color the X axis labels
    # p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1, facecolor='green', transform=fig.transFigure)
    # p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1, facecolor='red', transform=fig.transFigure)
    # fig.add_artist(p1)
    # fig.add_artist(p2)
    fileName = './results-storage/charts/score-barcharts/BarChartfor-' + '-' + columnSplitMethod + '-' + missingValueFillMethodName + "-" + classificationMethod + ".png"
    plt.savefig(fileName)
    plt.show()

    return fileName


def plot_learn_curve(X_train, y_train, X_test, classificationMethod, fill_model, columnSplitMethod='Default'):
    """
    使用了反射，可能较难理解，为不同的自定义分类方法，不同的数据填充方法，不同的分类方法打分
    :param X_train:
    :param y_train:
    :param classificationMethod:['M1','M2',"M3',"M4"] 自定义分类方法
    :param fill_model:['knn', 'rf', 'interpolate'] 填充方法
    :param columnSplitMethod: 数据分割方法
    :return:
    """

    scoreDict = {}

    # 使用反射对 8 种分类方法进行运行，并计算分数
    for class_model in classification_model_methods:
        print("*" * 10, "输出-> 分割方法 ", columnSplitMethod, '分类方法', classificationMethod,
              "--空值填充方法", fill_model, "--模型方法", class_model, " learn Curve", "*" * 10)
        # 返回预测值和模型
        _predicted, model = classifier_selection(class_model, X_train, y_train, X_test)
        plot_lc(X_train, y_train, model, classificationMethod, fill_model, class_model, columnSplitMethod)


def plot_lc(x, y, model, classificationMethod, fill_model, class_model, columnSplitMethod='Default'):
    """

    :param x:
    :param y:
    :param model:
    :param classificationMethod: 分类方法
    :param fill_model: 填充方法
    :param class_model: 模型
    :param columnSplitMethod: 数据分割方法
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
    ax.set_xlabel(
        '分割方法' + columnSplitMethod + '分类方法' + classificationMethod + "填充方法" + fill_model + "模型" + class_model)  # 设置子图的x轴名称
    ax.set_ylabel("score")
    ax.grid()  # 画出网图
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='train score')
    # 画训练集数据分数，横坐标为用作训练的样本数，纵坐标为不同折下的训练分数的均值
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='test score')
    ax.legend(loc='best')  # 设置图例
    plt.savefig(
        "./results-storage/charts/learn_curve/" + columnSplitMethod + "_" + classificationMethod + "_" + fill_model + "_" + class_model + "_learn_curve.png")
    plt.show()
