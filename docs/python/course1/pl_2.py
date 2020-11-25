
import pandas as pd
import numpy as np

data_old =pd.read_excel("./docs/python/course1/t_data/行业指数.xlsx",skiprows=1,skipfooter=2)

data = data_old.set_index("日期")

data_pct = data.pct_change().dropna()

data_rolling = data["全指金融"].rolling(250).mean().dropna()
data_std = data["全指金融"].rolling(250).std().dropna()


pe_data = pd.read_csv("./docs/python/course1/t_data/行业指数pe_ttm.CSV",encoding="gbk",skiprows=[0,],header=1)

print(pe_data.head())
print(pe_data["全指能源"].quantile(list(np.arange(11)/10)))

pe_data.set_index("时间",inplace=True)

df = pd.DataFrame()
for i in pe_data.columns:
    df1 = pd.DataFrame(pe_data[i].quantile(list(np.arange(11)/10)))
    df = pd.concat([df1,df],axis = 1)

df_now = pd.DataFrame(pe_data.iloc[0,:]).T.rename({"2019/3/29":"now"})

df = pd.concat([df_now,df],axis=0,sort=False)

label = ["1","2","3","4","5","6","7","8","9","10"]
qc = pd.qcut(data["全指医药"],10,labels=label)

df = pd.DataFrame()

for i in pe_data.columns:
    new_data = pe_data[i][pe_data[i]>0]
    df1= pd.DataFrame(pd.qcut(new_data,10,labels=label))
    df = pd.concat([df1,df],axis = 1,sort=False)

print(df.iloc[0,:])