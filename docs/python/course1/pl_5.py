import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


years = range(1880,2011)
pieces = []
columns = ["name","sex","births"]

for year in years:
    path ="./docs/python/course1/t_data/babynames/yob%d.txt"%year
    frame = pd.read_csv(path,names=columns)
    frame["year"]=year
    pieces.append(frame)

names = pd.concat(pieces,ignore_index=True)

total_births = names.pivot_table('births',index='year',columns='sex',aggfunc='sum')

# total_births.plot(title="根据性别和年份划分的出生人口")
# plt.show()

def add_prop(group):
    group["prop"]=group["births"]/group["births"].sum()
    return group

names = names.groupby(['year','sex']).apply(add_prop)

pieces = []
for year,group in names.groupby(['year','sex']):
    pieces.append(group.sort_values(by="births",ascending=False)[:1000])

top1000 = pd.concat(pieces,ignore_index=True)

table = top1000.pivot_table('prop',index='year',columns='sex',aggfunc='sum')
table.plot(title='前一千名男性女性名字分别的占比')

plt.show()

