import numpy as np
import pandas as pd

def test1():
    df = pd.DataFrame({'Class':[1,1,2,2],
                       'Name':['San Zhang','San Zhang','Si Li','Si Li'],
                       'Subject':['Chinese','Math','Chinese','Math'],
                       'Grade':[80,75,90,85]})
    
    print(df)
    print(df.pivot(index='Name',columns='Subject',values='Grade'))

    df = pd.DataFrame({'Class':[1, 1, 2, 2, 1, 1, 2, 2],
                      'Name':['San Zhang', 'San Zhang', 'Si Li', 'Si Li',
                               'San Zhang', 'San Zhang', 'Si Li', 'Si Li'],
                      'Examination': ['Mid', 'Final', 'Mid', 'Final',
                                     'Mid', 'Final', 'Mid', 'Final'],
                      'Subject':['Chinese', 'Chinese', 'Chinese', 'Chinese',
                                  'Math', 'Math', 'Math', 'Math'],
                      'Grade':[80, 75, 85, 65, 90, 85, 92, 88],
                      'rank':[10, 15, 21, 15, 20, 7, 6, 2]})
    
    print(df)
    print(df.pivot(index=['Class','Name'],columns=['Examination','Subject'],values=['Grade','rank']))

    df = pd.DataFrame({'Name':['San Zhang', 'San Zhang',
                               'San Zhang', 'San Zhang',
                               'Si Li', 'Si Li', 'Si Li', 'Si Li'],
                      'Subject':['Chinese', 'Chinese', 'Math', 'Math',
                                  'Chinese', 'Chinese', 'Math', 'Math'],
                      'Grade':[80, 90, 100, 90, 70, 80, 85, 95]})
    
    print(df)
    print(df.pivot_table(index='Name',columns='Subject',values='Grade',aggfunc='mean'))
    print(df.pivot_table(index='Name',columns='Subject',values='Grade',aggfunc= lambda x :x.mean()))
    print(df.pivot_table(index='Name',columns='Subject',values='Grade',aggfunc='mean',margins=True))

# test1()

def test2():
    df = pd.DataFrame({'Class':[1,2],
                      'Name':['San Zhang', 'Si Li'],
                      'Chinese':[80, 90],
                      'Math':[80, 75]})
    
    print(df)
    df_melted = df.melt(id_vars=['Class','Name'],value_vars=['Chinese','Math'],var_name='Subject',value_name='Grade')
    print(df_melted)

    df = pd.DataFrame({'Class':[1,2],'Name':['San Zhang', 'Si Li'],
                       'Chinese_Mid':[80, 75], 'Math_Mid':[90, 85],
                       'Chinese_Final':[80, 75], 'Math_Final':[90, 85]})
    
    print(df)
    print(pd.wide_to_long(df,stubnames=['Chinese','Math'],i=['Class','Name'],j='Examination',sep='_',suffix='.+'))


    df = pd.DataFrame({'Class':[1, 1, 2, 2, 1, 1, 2, 2],
                      'Name':['San Zhang', 'San Zhang', 'Si Li', 'Si Li',
                               'San Zhang', 'San Zhang', 'Si Li', 'Si Li'],
                      'Examination': ['Mid', 'Final', 'Mid', 'Final',
                                     'Mid', 'Final', 'Mid', 'Final'],
                      'Subject':['Chinese', 'Chinese', 'Chinese', 'Chinese',
                                  'Math', 'Math', 'Math', 'Math'],
                      'Grade':[80, 75, 85, 65, 90, 85, 92, 88],
                      'rank':[10, 15, 21, 15, 20, 7, 6, 2]})
    res = df.pivot(index = ['Class', 'Name'],
                           columns = ['Subject','Examination'],
                           values = ['Grade','rank'])

    print(res)
    res.columns = res.columns.map(lambda x:'_'.join(x))
    res = res.reset_index()

    res = pd.wide_to_long(res,stubnames=['Grade','rank'],i =['Class','Name'],j='Subject_Examination',sep='_',suffix='.+')
    res = res.reset_index()
    res[['Subject', 'Examination']] = res['Subject_Examination'].str.split('_', expand=True)
    res = res[['Class', 'Name', 'Examination','Subject', 'Grade', 'rank']].sort_values('Subject')
    res = res.reset_index(drop=True)
    print(res)

    df = pd.DataFrame(np.ones((4,2)),
                      index = pd.Index([('A', 'cat', 'big'),
                                        ('A', 'dog', 'small'),
                                        ('B', 'cat', 'big'),
                                        ('B', 'dog', 'small')]),
                      columns=['col_1', 'col_2'])

    print(df)
    print(df.unstack())
    print(df.unstack([0,2]))

    df = pd.DataFrame(np.ones((4,2)),
                      index = pd.Index([('A', 'cat', 'big'),
                                        ('A', 'dog', 'small'),
                                        ('B', 'cat', 'big'),
                                        ('B', 'dog', 'small')]),
                      columns=['index_1', 'index_2']).T

    print(df)
    print(df.stack())
    print(df.stack([1,2]))


# test2()

def test3():
    df = pd.read_csv('./docs/python/course3/data/drugs.csv').sort_values([
         'State','COUNTY','SubstanceName'],ignore_index=True)
    print(df.head())

    res = df.pivot(index=['State','COUNTY','SubstanceName'],columns='YYYY',values='DrugReports').reset_index()
    res = res.rename_axis(columns={'YYYY':''})
    print(res.head())

    res_1 = res.melt(id_vars=['State','COUNTY','SubstanceName'],value_vars=res.columns[-8:],var_name = 'YYYY',value_name='DrugReports')
    res_1 = res_1.dropna(subset=['DrugReports'])
    res_1 = res_1[df.columns].sort_values(['State','COUNTY','SubstanceName'],ignore_index=True).astype({'YYYY':'int64','DrugReports':'int64'})

    print(res_1.equals(df))

    res_2 = df.pivot_table(index='YYYY',columns='State',values='DrugReports',aggfunc='sum')

    print(res_2.head())

    res_3 = df.groupby(['State','YYYY'])['DrugReports'].sum().to_frame().unstack(0).droplevel(0,axis=1)

    print(res_3)



# test3()

# 从功能上看， melt 方法应当属于 wide_to_long 的一种特殊情况，即 stubnames 只有一类。
# 请使用 wide_to_long 生成 melt 一节中的 df_melted 。（提示：对列名增加适当的前缀）

def test4():
    df = pd.DataFrame({'Class':[1,2],
                      'Name':['San Zhang', 'Si Li'],
                      'Chinese':[80, 90],
                      'Math':[80, 75]})
    
    df = df.rename(columns={'Chinese':'pre_Chinese','Math':'pre_Math'})
    df = pd.wide_to_long(df,stubnames=['pre'],i=['Class','Name'],j='Subject',sep='_',suffix='.+')
    df = df.reset_index().rename(columns={'pre':'Grade'})
    print(df)
    
test4()

