#数据标准化

#等宽离散化
import pandas as pd
import numpy as np
import random

age = np.random.rand(10)
print(age)
data = pd.cut(age,3)
print(data)
data = pd.cut(age,3,labels=[1,2,3])
print(data)

#等频离散化

sample = pd.DataFrame({"normal":np.random.randn(10)})
data = pd.cut(sample.normal,bins=sample.normal.quantile([0,0.5,1]),include_lowest=True)
print(data)
data = pd.cut(sample.normal,bins=sample.normal.quantile([0,0.5,1]),include_lowest=True,labels=['good','bad'])
print(data)


