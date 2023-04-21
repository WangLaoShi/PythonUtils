# 回归算法在scikit-learn中的使用方法：
#
# 主要参考网页：https://ster.im/py_sklearn_1/
# 基础模型：
#
# * 线性回归（包含岭回归、Lasso回归、弹性网络回归）
# * 树回归
# * 支持向量机回归
# * K近邻回归
#
# 集成模型：
#
# * 随机森林回归
# * 极端随机树回归
# * AdaBoost回归
# * Gradient Boosting回归

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor



# 这个类使用了反射，比较难。
class RegressorCollection:
    @staticmethod
    def Linear(X_train,y_train,X_test,y_test=None):
        """
        最小二乘法线性回归
        最基本的线性回归法，它接收如下的几个参数：

        fit_intercept：是否考察截距项b，默认为True。
        normalize：是否先对数据进行Z-score标准化，默认为False。
        copy_X：默认为True则复制X，否则直接在原X上覆写。
        n_jobs：使用的处理器核数，默认None表示单核。
        """
        reg = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
        reg.fit(X_train, y_train)
        if y_test is None: # 如果为 None 的话，表示操作，否则是测试。
            predicted = reg.predict(X_test)
            return predicted,reg
        else:
            reg.score(X_test, y_test)  # 回归模型score返回的是R方，下同
            # 各特征的系数w
            print("各特征的系数w")
            print(reg.coef_)
            # 截距b
            print("截距b")
            print(reg.intercept_)
            return None,None

    @staticmethod
    def Ridge(X_train,y_train,X_test,y_test=None):
        """
        岭回归
        带L2正则项的线性回归，相比LinearRegression主要多一个正则项系数
        α
        的参数。

        与Ridge相比，RidgeCV内置了交叉验证，会自动帮我们筛出
        α
        的最优解，省去了超参数调试的麻烦，因此通常采用后者。
        """
        reg = RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True,
                      scoring=None, cv=5, gcv_mode=None, store_cv_values=False)
        reg.fit(X_train, y_train)
        if y_test is None:
            predicted = reg.predict(X_test)
            return predicted, reg
        else:
            reg.score(X_test, y_test)
            # 正则项系数alpha
            print("正则项系数alpha")
            print(reg.alpha_)
            return None,None

    @staticmethod
    def Lasso(X_train,y_train,X_test,y_test=None):
        """
        带L1正则项的线性回归，常用来估计稀疏参数的高维线性模型。

        供有Lasso、LassoCV、LassoLars、LassoLarsCV、LassoLarsIC五种可供选择，
        带CV的即自动选择最优的正则项系数，带Lars的采用最小角回归法而不带Lars的采用坐标轴
        下降法进行损失函数优化。LassoLarsIC采用AIC（Akaike信息准则）或BIC（Bayes信息准则）
        确定正则项系数。在大多数回归任务中，首选LassoCV，次选LassoLarsCV。
        """
        reg = LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True,
                      precompute="auto", max_iter=1000, tol=0.0001,
                      copy_X=True, cv=5, verbose=False, n_jobs=None,
                      positive=False, random_state=None, selection="cyclic")
        reg.fit(X_train, y_train)

        if y_test is None:
            predicted = reg.predict(X_test)
            return predicted, reg
        else:
            reg.score(X_test, y_test)
            return None,None

    @staticmethod
    def Elastic(X_train,y_train,X_test,y_test=None):

        """
        弹性网络回归
        同时带有L1和L2正则项的线性回归，使用l1_ratio这一权重参数来分配L1和L2正则项的比重。
        常用ElasticNetCV，它会自动选择正则项系数和平衡权重。
        """
        reg = ElasticNetCV(l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None,
                           fit_intercept=True, precompute="auto",
                           max_iter=1000, tol=0.0001, cv=5, copy_X=True, verbose=0,
                           n_jobs=None, positive=False, random_state=None, selection="cyclic")
        reg.fit(X_train, y_train)

        if y_test is None:
            predicted = reg.predict(X_test)
            return predicted, reg
        else:
            reg.score(X_test, y_test)
            return None, None

    @staticmethod
    def DecisionTree(X_train,y_train,X_test,y_test=None):
        """
        树回归
        CART用于回归时，参数与分类器类似，它可以接收如下的参数：

        criterion：分枝的标准，默认"mse"为均方差，可选"friedman_mse"（Friedman均方差）或者"mae"（绝对平均误差）。通常采用默认值。
        splitter：分枝的策略，默认"best"在所有划分点中找出最优的划分点，适合样本量不大的情况。样本量巨大时建议选择"random"，在部分划分点中找局部最优的划分点。
        max_depth：限制树的最大深度，默认值为None。如果样本和特征很多时可以适当限制树的最大深度。
        min_samples_split：分割一个节点所需的最小样本数，默认为2，当样本量非常大时可以增加这个值。
        min_samples_leaf：叶节点上所需的最小样本数，叶节点样本数少于这个值时会被剪枝。默认为1，当样本量非常大时可以增加这个值。
        min_weight_fraction_leaf：叶节点样本权重和所需的最小值，默认为0即视样本具有相同的权重。
        max_features：分枝时考虑的特征数量最大值，默认"auto"即该值等于特征数量。可以指定整数或者浮点数（表示占特征总数的比例）。也可选"sqrt"（特征数的开根）、"log2"（特征数的对数）、None（等于特征数）。如果特征数较多可以考虑限制以加快模型拟合。
        random_state：随机数种子。
        max_leaf_nodes：叶节点数最大值，默认None不对叶节点数量做限制，如果特征较多可以加以限制。
        min_impurity_decrease：默认为0.，如果分枝导致不纯度的减少大于等于该值，则节点将被分枝。
        min_impurity_split：默认为1e-7，如果某节点的不纯度超过这个阈值，则该节会分枝，否则该节点为叶节点。
        presort：是否对数据进行预排序，以加快寻找最佳分割点。默认为False。当使用小数据集或对深度作限制时，设置为True可能会加速训练，但对于大型数据集则反而会变慢。
        我们超参数调优的主要对象为max_depth、min_samples_split、min_samples_leaf、max_features。
        """
        # The 'criterion' parameter of DecisionTreeRegressor
        # must be a str among
        # {'friedman_mse', 'poisson', 'absolute_error', 'squared_error'}.
        # Got 'mse' instead.
        reg = DecisionTreeRegressor(criterion="friedman_mse", splitter="best", max_depth=None,
                                    min_samples_split=2, min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.0, max_features=None,
                                    random_state=None, max_leaf_nodes=None,
                                    min_impurity_decrease=0.0)
        reg.fit(X_train, y_train)

        if y_test is None:
            predicted = reg.predict(X_test)
            return predicted, reg
        else:
            reg.score(X_test, y_test)
            return None, None

    @staticmethod
    def SVR(X_train,y_train,X_test,y_test=None):
        """
        支持向量机回归
        部分参数如下：

        kernel：核函数，默认使用"rbf"径向基函数，可选"linear"、"poly"、"sigmoid"、"precomputed"或者一个可调用的函数。
        degree：多项式核函数的维度d，仅在核函数选择"poly"时有效。默认值为3。
        gamma："rbf"、"poly"、"sigmoid"的系数gamma，默认为"auto"，取特征数量的倒数。
        coef0：核函数中的独立项，仅在核函数选择"poly"、"sigmoid"时有效。默认值为0.0。
        tol：停止训练的误差精度，默认值为1e-3。
        C：惩罚系数C，默认值为1.0。
        max_iter：最大迭代次数，默认为-1即无限制。
        最重要的两个调参对象是gamma和C。gamma越大，支持向量越少，gamma越小，支持向量越多。C可理解为逻辑回归中正则项系数lambda的倒数，C过大容易过拟合，C过小容易欠拟合。通常采用网格搜索法进行调参。
        """
        reg = SVR(kernel="rbf", degree=3, gamma="auto", coef0=0.0,
                  tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
                  cache_size=200, verbose=False, max_iter=-1)
        reg.fit(X_train, y_train)

        if y_test is None:
            predicted = reg.predict(X_test)
            return predicted, reg
        else:
            reg.score(X_test, y_test)
            return None, None

    @staticmethod
    def KNN(X_train,y_train,X_test,y_test=None):
        """
        K近邻回归
        部分参数如下：

        n_neighbors：最近邻单元的个数K。
        weights：是否考虑邻居的权重，默认值"uniform"视每个邻居的权重相等，"distance"则给较近的单元更大的权重（取距离的倒数），也可以指定一个可调用的函数。
        algorithm：计算最近邻的算法，默认"auto"自动挑选模型认为最合适的，可选"ball_tree"、"kd_tree"、"brute"。
        leaf_size：叶节点数量，默认值30，只有在algorithm选择球树或者KD树时有效。
        p：闵式距离的度量，p=1时为曼哈顿距离，p=2时为欧式距离（默认）。
        n_neighbors是最需要关注的超参数，其次weights和p也可以适当调整。
        """
        reg = KNeighborsRegressor(n_neighbors=5, weights="uniform", algorithm="auto",
                                  leaf_size=30, p=2, metric="minkowski", metric_params=None)
        reg.fit(X_train, y_train)
        if y_test is None:
            predicted = reg.predict(X_test)
            return predicted, reg
        else:
            reg.score(X_test, y_test)
            return None, None

    @staticmethod
    def Bagging(X_train,y_train,X_test,y_test=None):
        """
        集成回归模型：Bagging
        Bagging回归
        参数：

        base_estimator：基模型，默认None代表决策树，可选择其它基础回归模型对象。
        n_estimators：基模型的数量，默认为10。
        max_samples：用于训练基模型的从X_train中抽取样本的数量，可以是整数代表数量，也可以是浮点数代表比例，默认为1.0。
        max_features：用于训练基模型的从X_train中抽取特征的数量，可以是整数代表数量，也可以是浮点数代表比例，默认为1.0。
        bootstrap：对于样本是否有放回抽样，默认为True。
        bootstrap_features：对于特征是否有放回抽样，默认为False。
        oob_score：是否使用包外样本估计泛化误差。
        warm_start：默认为False，如果选择True，下一次训练以上一次模型的参数为初始参数。
        对于所有的集成模型，最需要关注的超参数是n_estimators，即基模型的数量，通常需要使用网格搜索法寻找最优解；其他的参数通常保持默认即可取得较好的效果。
        """
        reg = BaggingRegressor(base_estimator=None, n_estimators=10, max_samples=1.0,
                               max_features=1.0, bootstrap=True, bootstrap_features=False,
                               oob_score=False, warm_start=False, random_state=None, verbose=0)
        reg.fit(X_train, y_train)
        if y_test is None:
            predicted = reg.predict(X_test)
            return predicted, reg
        else:
            reg.score(X_test, y_test)
            return None, None

    @staticmethod
    def RF(X_train,y_train,X_test,y_test=None):
        """
        随机森林回归
        参数：

        n_estimators：树的数量，默认为10。
        criterion：分枝的标准，默认"mse"为均方差，可选"mae"（绝对平均误差）。
        max_depth：限制树的最大深度，默认值为None，表示一直分枝直到所有叶节点都是纯的，或者所有叶节点的样本数小于min_samples_split。
        min_samples_split：分割一个节点所需的最小样本数，默认为2。
        min_samples_leaf：叶节点上所需的最小样本数，叶节点样本数少于这个值时会被剪枝。默认为1。
        min_weight_fraction_leaf：叶节点样本权重和所需的最小值，默认为0即视样本具有相同的权重。
        max_features：分枝时考虑的特征数量最大值，默认"auto"即该值等于特征数量。可以指定整数或者浮点数（表示占特征总数的比例）。也可选"sqrt"（特征数的开根）、"log2"（特征数的对数）、None（等于特征数）。
        max_leaf_nodes：叶节点数最大值，默认None不对叶节点数量做限制。
        min_impurity_decrease：默认为0，如果分枝导致不纯度的减少大于等于该值，则节点将被分枝。
        min_impurity_split：默认为1e-7，如果某节点的不纯度超过这个阈值，则该节会分枝，否则该节点为叶节点。
        bootstrap：对于样本是否有放回抽样，默认为True。如果为False，则使用整个数据集构建每个树。
        oob_score：是否使用包外样本估计R方。默认为False。
        random_state：随机数种子。
        warm_start：默认为False，如果选择True，下一次训练以上一次模型的参数为初始参数。
        除了n_estimators之外，还可以考虑适当调整max_depth、min_samples_split、min_samples_leaf、max_features这些决策树的参数。
        """
        # The 'criterion' parameter of RandomForestRegressor
        # must be a str among {'friedman_mse', 'poisson', 'absolute_error', 'squared_error'}. Got 'mse' instead.
        reg = RandomForestRegressor(n_estimators=10, criterion="friedman_mse", max_depth=None,
                                    min_samples_split=2, min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.0, max_features="auto",
                                    max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    bootstrap=True, oob_score=False,
                                    random_state=None, verbose=0, warm_start=False)
        reg.fit(X_train, y_train)
        if y_test is None:
            predicted = reg.predict(X_test)
            return predicted, reg
        else:
            reg.score(X_test, y_test)
            # 各特征的重要性
            reg.feature_importances_
            print("# 各特征的重要性")
            print(reg.feature_importances_)
            return None, None

    @staticmethod
    def ExtraTree(X_train,y_train,X_test,y_test=None):
        """
        极端随机树回归
        Extra Tree和随机森林的区别较小，参数几乎一致。
        """
        #  The 'criterion' parameter of ExtraTreesRegressor must be a str among
        #  {'poisson', 'squared_error', 'absolute_error', 'friedman_mse'}. Got 'mse' instead.
        reg = ExtraTreesRegressor(n_estimators=10, criterion="friedman_mse", max_depth=None,
                                  min_samples_split=2, min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0, max_features="auto",
                                  max_leaf_nodes=None, min_impurity_decrease=0.0,
                                  bootstrap=False, oob_score=False,
                                  random_state=None, verbose=0, warm_start=False)
        reg.fit(X_train, y_train)
        if y_test is None:
            predicted = reg.predict(X_test)
            return predicted, reg
        else:
            reg.score(X_test, y_test)
            return None, None

    @staticmethod
    def ADA(X_train,y_train,X_test,y_test=None):
        """
        AdaBoost回归
        参数：

        base_estimator：弱回归学习器，可指定为任意回归模型对象，默认为None，即DecisionTreeRegressor（max_depth=3）。
        n_estimators：最大迭代次数，即弱学习器的最大个数，默认为50。
        learning_rate：每个弱学习器的权重缩减系数，介于0.和1.之间，默认为1.。
        loss：每次迭代后更新权重时采用的损失函数，默认为"linear"，可选"square"、"exponential"，通常使用默认值。
        random_state：随机数种子。
        n_estimators和learning_rate两个参数相互牵制，通常会一起进行调参。
        """
        reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=3), n_estimators=50,
                                learning_rate=1.0, loss="linear")
        reg.fit(X_train, y_train)
        if y_test is None:
            predicted = reg.predict(X_test)
            return predicted, reg
        else:
            reg.score(X_test, y_test)
            return None, None

    @staticmethod
    def GB(X_train,y_train,X_test,y_test=None):
        """
        Gradient Boosting回归
        其中决策树部分的参数不列举。

        loss：损失函数，默认值"ls"代表最小二乘回归，可选"lad"（最小绝对偏差）、"huber"（前两者的结合）和"quantile"（分位数回归）。
        learning_rate：每棵树的权重缩减系数，默认为0.1，与n_estimators相互牵制，是调参的重点。
        n_estimators：最大迭代次数，默认为100。
        subsample：子采样率，用于训练每棵树的样本占样本总数的比例，默认为1.0，如使用小于1.0的值，该模型就为随机梯度提升，会减少方差、增大偏差。
        init：默认为None，可指定具有fit和predict方法的预测器对象，它用于初始化参数。
        """
        # The 'loss' parameter of GradientBoostingRegressor must be a str among
        # {'squared_error', 'huber', 'quantile', 'absolute_error'}. Got 'ls' instead.
        reg = GradientBoostingRegressor(loss="squared_error", learning_rate=0.1, n_estimators=100,
                                        subsample=1.0, criterion="friedman_mse", min_samples_split=2,
                                        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
                                        min_impurity_decrease=0.0, init=None,
                                        random_state=None, max_features=None, alpha=0.9, verbose=0,
                                        max_leaf_nodes=None, warm_start=False,
                                        validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
        reg.fit(X_train, y_train)
        if y_test is None:
            predicted = reg.predict(X_test)
            return predicted, reg
        else:
            reg.score(X_test, y_test)
            return None, None

