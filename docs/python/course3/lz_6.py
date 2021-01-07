import pandas as pd
import numpy as np

def test1():
    df =pd.read_csv("./docs/python/course3/data/learn_pandas.csv")
    print(df.head())
    # print(df.groupby(["School","Gender"])["Height"].median())

    condition = df.Weight>df.Weight.mean()
    # print(df.groupby(condition)["Height"].mean())

    df["WeightClass"] = df["Weight"].mask(df["Weight"]>df["Weight"].quantile(0.75),"high").mask(df["Weight"]<df["Weight"].quantile(0.25),"low").mask((df["Weight"]<=df["Weight"].quantile(0.75)) & (df["Weight"]>=df["Weight"].quantile(0.25)),"normal")

    print(df.groupby("WeightClass")["Height"].mean())

    item = np.random.choice(list('abc'),df.shape[0])

    # print(df.groupby(item)['Height'].mean())
    # print(df.groupby([condition,item])['Height'].mean())

    df[['School','Gender']].drop_duplicates()
    print(df.groupby([df['School'],df['Gender']])['Height'].mean())

    gb = df.groupby(['School','Grade'])
    print(gb.ngroups)
    print(gb.groups.keys())
    print(gb.size())


# test1()

df =pd.read_csv("./docs/python/course3/data/learn_pandas.csv")

def test2():
    # df =pd.read_csv("./docs/python/course3/data/learn_pandas.csv")
    print(df.head())

    gb = df.groupby('Gender')['Height']
    print(gb.idxmin())
    print(gb.quantile(0.95))

    gb =df.groupby('Gender')[['Height','Weight']]

    print(gb.max())

    print(gb.agg(['sum','idxmax','skew']))
    print(gb.agg({'Height':['mean','max'],'Weight':'count'}))
    print(gb.agg(lambda x : x.mean()-x.min()))
    print(gb.describe())
    print(gb.agg(my_func))
    print(gb.agg([('range',lambda x:x.max()-x.min()),('my_sum','sum')]))

    print(gb.cummax().head())
    print(gb.transform(lambda x:(x-x.mean())/x.std()).head())
    print(gb.transform("mean").head())
    print(gb.filter(lambda x:x.shape[0]>100).head())



def my_func(s):
    res='High'
    if s.mean() <= df[s.name].mean():
        res ='Low'
    return res


# test2()


def test3():
    gb =df.groupby('Gender')[['Height','Weight']]
    print(gb.apply(BMI))

    print(df.head())
    pass


def BMI(x):
    Height = x['Height']/100
    Weight = x['Weight']
    BMI_value = Weight/Height**2
    return BMI_value.mean()

# test3()


def test4():
    df =pd.read_csv("./docs/python/course3/data/car.csv")
    print(df.head())

    df_1 = df.groupby('Country').filter(lambda x:x.shape[0]>2)
    print(df_1.groupby('Country')['Price'].agg([('Cov',lambda x:x.std()/x.mean()),'mean','count']))

    df['g']= 'second'
    df.loc[:int(df.index.max()/3),'g'] = 'first'
    df.loc[int(df.index.max()*2/3):,'g'] = 'third'

    print(df.groupby('g')['Price'].mean())

    df_2 = df.groupby('Type').agg({'Price':['max','min'],'HP':['max','min']})

    df_2.columns = df_2.columns.map(lambda x :x[0]+"_"+x[1])
    print(df_2)

    print(df.groupby('Type')['HP'].transform(normalize).head())

    print(df.groupby('Type')[['HP','Disp.']].apply(lambda x: np.corrcoef(x['HP'].values,x['Disp.'].values)[0,1]))

    
def normalize(s):
    s_min,s_max=s.min(),s.max()
    return (s - s_min) / (s_max - s_min)


# test4()


# Ex2：实现transform函数
#     groupby 对象的构造方法是 my_groupby(df, group_cols)
#     支持单列分组与多列分组
#     支持带有标量广播的 my_groupby(df)[col].transform(my_func) 功能
#     pandas 的 transform 不能跨列计算，请支持此功能，即仍返回 Series 但 col 参数为多列
#     无需考虑性能与异常处理，只需实现上述功能，在给出测试样例的同时与 pandas 中的 transform 对比结果是否一致

