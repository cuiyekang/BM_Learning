import pandas as pd
import numpy as np

# df =pd.read_csv("./docs/python/course3/data/learn_pandas.csv")

# # print(df.columns)
# df = df[df.columns[:7]]
# print(df.head(2))
# print(df.tail(3))
# print(df.info())
# print(df.describe())

# df_demo = df[["Height","Weight"]]
# print(df_demo.mean())
# print(df_demo.max())
# print(df_demo.quantile(0.75))
# print(df_demo.count())
# print(df_demo.idxmax())
# print(df_demo.mean(axis=1).head())
# print(df["School"].unique())
# print(df["School"].nunique())
# print(df["School"].value_counts())


# df_demo = df[['Gender','Transfer','Name']]
# print(df_demo.drop_duplicates(['Gender','Transfer']))
# print(df_demo.drop_duplicates(['Gender','Transfer'],keep="last"))
# print(df_demo.drop_duplicates(['Name','Gender'],keep=False).head())
# print(df["School"].drop_duplicates())


# 现有一份口袋妖怪的数据集，下面进行一些背景说明：
#     # 代表全国图鉴编号，不同行存在相同数字则表示为该妖怪的不同状态
#     妖怪具有单属性和双属性两种，对于单属性的妖怪， Type 2 为缺失值
#     Total, HP, Attack, Defense, Sp. Atk, Sp. Def, Speed 分别代表种族值、体力、物攻、防御、特攻、特防、速度，其中种族值为后6项之和

# 对 HP, Attack, Defense, Sp. Atk, Sp. Def, Speed 进行加总，验证是否为 Total 值。
# 对于 # 重复的妖怪只保留第一条记录，解决以下问题：
# 求第一属性的种类数量和前三多数量对应的种类
# 求第一属性和第二属性的组合种类
# 求尚未出现过的属性组合
# 按照下述要求，构造 Series ：
# 取出物攻，超过120的替换为 high ，不足50的替换为 low ，否则设为 mid
# 取出第一属性，分别用 replace 和 apply 替换所有字母为大写
# 求每个妖怪六项能力的离差，即所有能力中偏离中位数最大的值，添加到 df 并从大到小排序

def test1():

    df =pd.read_csv("./docs/python/course3/data/pokemon.csv")

    df = df[:50]

    df["cal_total"] = df[df.columns[5:]].sum(axis=1)

    df_one = df.drop_duplicates(["#"])

    # print(df_one["Type 1"].nunique())
    # print(df_one["Type 1"].value_counts().head(3))

    df_two = df_one.drop_duplicates(["Type 1","Type 2"])
    df_two = df_two.fillna("NaN")

    # print(df_two[["Type 1","Type 2"]])

    all_first = df_two["Type 1"].unique()
    all_second = df_two["Type 2"].unique()

    np.append(all_second,"NaN")
    not_have =[] 

    for first in all_first:
        for second in all_second:
            if df_two[df_two["Type 1"] == first][df_two["Type 2"] == second]["Type 1"].count() == 0:
                not_have.append("{},{}".format(first,second))

    # print(not_have)
    # df["Attack"] = df["Attack"].mask(df["Attack"]>120,"high").mask(df["Attack"]<50,"low").mask((df["Attack"]>=50) & (df["Attack"]<=120),"mid")
    
    s_lower = list(df["Type 1"].unique())
    s_upper = np.char.upper(s_lower)

    # df["Type 1"] = df["Type 1"].replace(s_lower,s_upper)
    df["Type 1"] = df["Type 1"].apply(lambda x:x.upper())

    df_mad = df[["HP","Attack" ,"Defense","Sp. Atk","Sp. Def" ,"Speed"]]
    df["six_mad"] = df_mad.mad(axis=1)

    df = df.sort_values("six_mad",ascending=False)

    print(df)
    

# test1()

def test2():
    np.random.seed(0)
    s = pd.Series(np.random.randint(-1,2,20).cumsum())
    a = s.ewm(alpha=0.2).mean()
    print(a)
    alpha = 0.2
    win = 10
    b1 = pd.Series(s[0])
    b2 = pd.Series([1])
    b_r = pd.Series(s[0])
    for index in s.index:
        if index == 0:
            continue
        b2[index] = (1-alpha) ** index
        b1[index] = s[index]
        
        b_temp = b2.sort_index(ascending=False).reset_index(drop=True)
        if index > win:
            start = index - win
            b_r[index] = sum(b1[start:]*b_temp[start:])/sum(b_temp[start:])
        else:
            b_r[index] = sum(b1*b_temp)/sum(b_temp)

    print(b_r)


# test2()

def doo(lst,alpha):
    w = pd.Series([0])
    if(len(lst) == 1):
        return lst[0]
    for i in range(len(lst)):
        w[i] = (1-alpha) ** i
    w = w.sort_index(ascending=False).reset_index(drop=True)
    return sum(lst * w) / sum(w)

    

def test3():
    np.random.seed(0)
    s = pd.Series(np.random.randint(-1,2,10).cumsum())
    a = s.ewm(alpha=0.2).mean()
    print(s)
    alpha = 0.2
    win = 3
    if len(s)>win:
        b = s.rolling(win).apply(lambda lst : doo(lst,alpha),raw = True)
    else:
        b = s.expanding().apply(lambda lst : doo(lst,alpha))
    print(b)

    

test3()