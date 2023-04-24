# 导入包,无需pip install
import pickle
import joblib

def dump_model(model, modelFileName):
    """
    将模型保存起来
    :param model:
    :param modelFileName:
    :return:
    """
    # 保存模型,我们想要导入的是模型本身，所以用“wb”方式写入，是二进制方式,DT是模型名字
    pickle.dump(model, open("./models_dump/pickle_" + modelFileName,
                            "wb"))  # open("dtr.dat","wb")意思是打开叫"dtr.dat"的文件,操作方式是写入二进制数据
    # 保存模型
    joblib.dump(model, './models_dump/joblib_' + modelFileName)  # 第二个参数只需要写文件名字,是不是比pickle更人性化


def load_model(modelFileName):
    """
    根据模型模型名，加载保存的模型
    :param modelFileName:
    :return:pickle，joblib
    """
    # 加载模型
    loaded_model = pickle.load(open("./models_dump/pickle_" + modelFileName, "rb"))
    # 加载模型
    loaded_model2 = joblib.load("./models_dump/joblib_" + modelFileName)

    return loaded_model, loaded_model2
