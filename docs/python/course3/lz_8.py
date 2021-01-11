import numpy as np
import pandas as pd


def test1():
    df1 = pd.DataFrame({'Name':['San Zhang','Si Li'],
                        'Age':[20,30]})
    df2 = pd.DataFrame({'Name':['Si Li','Wu Wang'],
                        'Gender':['F','M']})

    
    print(df1.merge(df2,on='Name',how='left'))

    df1 = pd.DataFrame({'df1_name':['San Zhang','Si Li'],
                        'Age':[20,30]})

    df2 = pd.DataFrame({'df2_name':['Si Li','Wu Wang'],
                        'Gender':['F','M']})    

    print(df1.merge(df2,left_on='df1_name',right_on='df2_name',how='left'))       

    df1 = pd.DataFrame({'Name':['San Zhang'],'Grade':[70]})
    df2 = pd.DataFrame({'Name':['San Zhang'],'Grade':[80]})

    print(df1.merge(df2,on='Name',how='left',suffixes=['_Chinese','_Math']))

    df1 = pd.DataFrame({'Name':['San Zhang', 'San Zhang'],
                        'Age':[20, 21],
                        'Class':['one', 'two']})

    df2 = pd.DataFrame({'Name':['San Zhang', 'San Zhang'],
                        'Gender':['F', 'M'],
                        'Class':['two', 'one']})

    print(df1.merge(df2,on=['Name','Class'],how='left'))



# test1()

def test2():
    df1 = pd.DataFrame({'Age':[20,30]},
                        index=pd.Series(
                        ['San Zhang','Si Li'],name='Name'))
    df2 = pd.DataFrame({'Gender':['F','M']},
                        index=pd.Series(
                        ['Si Li','Wu Wang'],name='Name'))        

    print(df1.join(df2,how='left'))

    df1 = pd.DataFrame({'Grade':[70]},
                        index=pd.Series(['San Zhang'],
                        name='Name'))

    df2 = pd.DataFrame({'Grade':[80]},
                        index=pd.Series(['San Zhang'],
                        name='Name'))

    print(df1.join(df2,how='left',lsuffix='_Chinese',rsuffix='_Math'))

    df1 = pd.DataFrame({'Age':[20,21]},
                        index=pd.MultiIndex.from_arrays(
                        [['San Zhang', 'San Zhang'],['one', 'two']],
                        names=('Name','Class')))

    df2 = pd.DataFrame({'Gender':['F', 'M']},
                        index=pd.MultiIndex.from_arrays(
                        [['San Zhang', 'San Zhang'],['two', 'one']],
                        names=('Name','Class')))

    print(df1.join(df2))
    


# test2()            

def test3():
    df1 = pd.DataFrame({'Name':['San Zhang','Si Li'],
                        'Age':[20,30]})
    df2 = pd.DataFrame({'Name':['Wu Wang'], 'Age':[40]})

    print(pd.concat([df1,df2]))                  

    df2 = pd.DataFrame({'Grade':[80, 90]})
    df3 = pd.DataFrame({'Gender':['M', 'F']})

    print(pd.concat([df1,df2,df3],1))  

    df2 = pd.DataFrame({'Name':['Wu Wang'], 'Gender':['M']})

    print(pd.concat([df1,df2]))

    s = pd.Series(['Wu Wang', 21], index = df1.columns)

    print(df1.append(s,ignore_index=True))

    s = pd.Series([80,90])
    print(df1.assign(Grade=s))
    df1['Grade']=s
    print(df1)

# test3()


def test4():
    df1 = pd.DataFrame({'Name':['San Zhang', 'Si Li', 'Wu Wang'],
                            'Age':[20, 21 ,21],
                            'Class':['one', 'two', 'three']})
    df2 = pd.DataFrame({'Name':['San Zhang', 'Li Si', 'Wu Wang'],
                            'Age':[20, 21 ,21],
                            'Class':['one', 'two', 'Three']}) 

    print(df1)
    print(df2)
    print(df1.compare(df2))
    print(df1.compare(df2,keep_shape=True))

# test4()

def test5():
    date = pd.date_range('20200412', '20201116').to_series()
    date = date.dt.month.astype('string').str.zfill(2) +'-'+ date.dt.day.astype('string').str.zfill(2) +'-'+ '2020'
    date = date.tolist()
    print(date[:5])

    df = pd.DataFrame(columns=['Date','Confirmed', 'Deaths', 'Recovered', 'Active'])

    for f in date:
        fp = "./docs/python/course3/data/us_report/"+f+".csv"
        one = pd.read_csv(fp)
        one = one[one['Province_State'] == 'New York'].loc[:,['Confirmed', 'Deaths', 'Recovered', 'Active']]
        one['Date'] = f
        df = pd.concat([df,one])

    print(df.set_index('Date'))    

test5()


# 请实现带有 how 参数的 join 函数
# 假设连接的两表无公共列
# 调用方式为 join(df1, df2, how="left")
# 给出测试样例
