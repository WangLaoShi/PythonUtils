import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas_util.feature_analytis import *

datas = pd.read_csv('../pandas_util/datas/second_hand_ house.csv')
# print(datas)

housePrice = datas['房屋单价']

feature_distribution(housePrice)