
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

pe_data = pd.read_excel("./docs/python/course1/t_data/pe_data.xlsx")
pe_data.dropna(inplace=True)
pe_data.drop(columns=["pb"],inplace=True)

data_grouped = pe_data.groupby(["date"])

def filter(group):
    group = group[(group['pe']>0) & (group['pe']<group['pe'].quantile(0.95))]
    return group.mean()

pe_ttm_mean = data_grouped.apply(filter)

q_arr = pe_ttm_mean.quantile(list(np.arange(11)/10)).values

q_list =[]

for i in range(len(q_arr)):
    q_list.append(q_arr[i][0])

pe_ttm_mean.plot(color='k',title='全A等权PE',rot=90,figsize=(10,6),fontsize=12,linewidth=2)
font = {"weight":"normal","size":12}
plt.xlabel("month",font)
plt.ylabel("等权PE",font)
plt.xticks(ticks=pd.date_range(pe_ttm_mean.index[0],pe_ttm_mean.index[-1],freq="Y"))
plt.hlines(q_list[5:],pe_ttm_mean.index[0],pe_ttm_mean.index[-1],color='r',linestyle='dashed',linewidth=1)
plt.hlines(q_list[:5],pe_ttm_mean.index[0],pe_ttm_mean.index[-1],color='g',linestyle='dashed',linewidth=1)
plt.show()
