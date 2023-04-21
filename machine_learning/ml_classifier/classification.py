import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

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


import numpy as np


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




