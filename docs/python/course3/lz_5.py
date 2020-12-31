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

    

# test3()

def test4():
    df_sample = pd.DataFrame({"id":list("abcde"),'value':[1,2,3,4,90]})

    # print(df_sample.sample(3,replace=True,weights = df_sample.value))

    df =pd.read_csv("./docs/python/course3/data/learn_pandas.csv")

    np.random.seed(0)

    multi_index = pd.MultiIndex.from_product([list("ABCD"),df.Gender.unique()],names=("School","Gender"))
    multi_column = pd.MultiIndex.from_product([["Heigh","Weight"],df.Grade.unique()],names=("Indicator","Grade"))

    df_multi = pd.DataFrame(np.c_[(np.random.randn(8,4)*5 + 163).tolist(),
                                (np.random.randn(8,4)*5 + 65).tolist()],
                            index=multi_index,columns = multi_column).round(1)

    # print(df_multi)

    df_multi = df.set_index(["School","Grade"])
    df_multi = df_multi.sort_index()

    res1 = df_multi.loc[(['Peking University', 'Fudan University'],['Sophomore', 'Junior']), :]
    res2 = df_multi.loc[[('Peking University', 'Junior'),('Fudan University','Sophomore')], :]

    print(res1)
    print(res2)




# test4()

def test5():
    np.random.seed(0)

    L1,L2 = ['A','B','C'],['a','b','c']
    mul_index1 = pd.MultiIndex.from_product([L1,L2],names=('Upper', 'Lower'))

    L3,L4 = ['D','E','F'],['d','e','f']
    mul_index2 = pd.MultiIndex.from_product([L3,L4],names=('Big', 'Small'))

    df_ex = pd.DataFrame(np.random.randint(-9,10,(9,9)),index = mul_index1,columns = mul_index2)

    print(df_ex)

    idx =pd.IndexSlice
    print(df_ex.loc[idx['C':, ('D', 'f'):]])
    
    print(df_ex.loc[idx[:'A', lambda x:x.sum()>0]]) # 列和大于0

    print(df_ex.loc[idx[:'A', 'b':], idx['E':, 'e':]])


# test5()

def test6():
    np.random.seed(0)

    L1,L2,L3 = ['A','B'],['a','b'],['alpha','beta']

    mul_index1 = pd.MultiIndex.from_product([L1,L2,L3],names=("Upper","Lower","Extra"))

    L4,L5,L6 = ['C','D'],['c','d'],['cat','dog']

    mul_index2 = pd.MultiIndex.from_product([L4,L5,L6],names=("Big","Small","Other"))

    df_ex = pd.DataFrame(np.random.randint(-9,10,(8,8)),index = mul_index1,columns=mul_index2)

    print(df_ex)

    print(df_ex.swaplevel(0,2,axis=1).head())
    print(df_ex.reorder_levels([2,0,1],axis=0).head())
    print(df_ex.droplevel(1,axis=1))
    print(df_ex.droplevel([0,1],axis=0))

    df_ex1 = df_ex.rename_axis(index={"Upper":"Changed_row"},columns={"Other":"Changed_col"})
    print(df_ex1)
    df_ex2 = df_ex.rename(columns={"cat":"not_cat"},level=2)
    print(df_ex2)
    df_ex3 = df_ex.rename(index=lambda x:str.upper(x),level=2)
    print(df_ex3)
    # df_ex3_1 = df_ex.rename_axis(index=lambda x:str.upper(x))
    # print(df_ex3_1)
    new_values = iter(list("abcdefgh"))
    df_ex4 = df_ex.rename(index=lambda x:next(new_values),level=2)
    print(df_ex4)

    df_temp = df_ex.copy()
    new_idx = df_temp.index.map(lambda x:(x[0],x[1],str.upper(x[2])))
    # df_temp.index = new_idx
    new_idx1 = df_temp.index.map(lambda x:(x[0] + "-"+x[1]+"-"+x[2]))
    df_temp.index=new_idx1
    new_idx2 = df_temp.index.map(lambda x:tuple(x.split("-")))
    df_temp.index = new_idx2

    print(df_temp)

# test6()

def test7():
    df = pd.DataFrame({'A':list('aacd'),'B':list('PQRT'),'C':[1,2,3,4]})
    print(df)
    print(df.set_index("A"))
    print(df.set_index("A",append=True))
    print(df.set_index(["A","B"]))
    my_index = pd.Series(list("WXYZ"),name="D")
    print(df.set_index(["A",my_index]))
    df_new = df.set_index(["A",my_index])
    print(df_new.reset_index("D"))
    print(df_new.reset_index("D",drop=True))
    print(df_new.reset_index())
    df_reindex = pd.DataFrame({"Weight":[60,70,80],"Height":[176,180,179]},index=['1001','1003','1002'])
    df_reindex = df_reindex.reindex(index=['1001','1002','1003','1004'],columns=['Weight','Gender'])
    print(df_reindex)
    df_existed = pd.DataFrame(index=['1001','1002','1003','1004'],columns=['Weight','Gender'])
    print(df_reindex.reindex_like(df_existed))

# test7()

def test8():
    df_set_1 = pd.DataFrame([[0,1],[1,2],[3,4]],index = pd.Index(['a','b','a'],name='id1'))
    df_set_2 = pd.DataFrame([[4,5],[2,6],[7,1]],index = pd.Index(['b','b','c'],name='id2'))

    id1,id2 = df_set_1.index.unique(),df_set_2.index.unique()
    # print(df_set_1)
    # print(df_set_2)
    print(id1.intersection(id2))
    print(id1.union(id2))
    print(id1.difference(id2))
    print(id1.symmetric_difference(id2))

    print(id1 & id2)
    print(id1 | id2)
    print((id1 ^ id2) & id1) 
    print(id1 ^ id2)

    df_set_in_col_1 = df_set_1.reset_index()
    df_set_in_col_2 = df_set_2.reset_index()

    print(df_set_in_col_1)
    print(df_set_in_col_2)
    print(df_set_in_col_1[df_set_in_col_1.id1.isin(df_set_in_col_2.id2)])



# test8()

def test9():
    df = pd.read_csv("./docs/python/course3/data/Company.csv")
    print(df.query('age<40 & (department == "Dairy" | department =="Bakery") & gender == "M"'))

    condition_1 = df.age<40 
    condition_2 = df.gender == "M"
    condition_3 = df.department == "Dairy" 
    condition_4 = df.department == "Bakery"
    condition_5 = condition_3 | condition_4
    conditoin_6 = condition_1 & condition_2 & condition_5

    # print(df.loc[conditoin_6])

    # print(df.iloc[1:-1:2,[0,2,-2]])

    df_1 = df.set_index(['department','job_title','gender'])
    df_1 = df_1.swaplevel(0,2,axis=0)
    df_1 = df_1.rename_axis(index={'gender':'Gender'})
    df_1.index =df_1.index.map(lambda x:(x[0] + '-' +x[1] + '-' + x[2]))
    df_1.index = df_1.index.map(lambda x:tuple(x.split('-')))
    df_1.index.names = ['gender','job_title','department']
    df_1 = df_1.reset_index([0,1,2])

    cols = list(df_1.columns)
    cols.append(cols.pop(2))
    cols.append(cols.pop(1))
    cols.append(cols.pop(0))

    df_1 = df_1[cols]

    print(df_1)
    


# test9()



# 把列索引名中的 \n 替换为空格。
# 巧克力 Rating 评分为1至5，每0.25分一档，请选出2.75分及以下且可可含量 Cocoa Percent 高于中位数的样本。
# 将 Review Date 和 Company Location 设为索引后，选出 Review Date 在2012年之后且 Company Location 不属于 France, Canada, Amsterdam, Belgium 的样本。

def test10():
    df = pd.read_csv("./docs/python/course3/data/chocolate.csv")
    print(df.head())

    df.columns = df.columns.map(lambda x :x.replace('\n',' '))

    
    condition_1 = df["Rating"] <=2.75
    df["Cocoa Percent"] =df["Cocoa Percent"].apply(lambda x : float(x.replace('%','')) / 100)
    condition_2 = df["Cocoa Percent"] > df["Cocoa Percent"].median()

    # df = df[condition_1 & condition_2]

    df_1 = df.set_index(['Review Date','Company Location'])
    df_1 = df_1.sort_index()

    idx = pd.IndexSlice

    df_1 = df_1.loc[idx[2012:,~df_1.index.get_level_values(1).isin(['France', 'Canada', 'Amsterdam', 'Belgium']),:]]
    print(df_1)




test10()