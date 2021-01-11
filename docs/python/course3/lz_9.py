import pandas as pd
import numpy as np

def test1():
    df = pd.read_csv('./docs/python/course3/data/learn_pandas.csv',
                     usecols = ['Grade', 'Name', 'Gender', 'Height',
                                'Weight', 'Transfer'])
    print(df.isna().mean())
    print(df[df.Height.isna()].head())

    sub_set = df[['Height', 'Weight', 'Transfer']]
    print(df[sub_set.isna().all(1)])
    print(df[sub_set.isna().any(1)].head())
    print(df[sub_set.notna().all(1)].head())

    res = df.dropna(how='any',subset=['Height', 'Weight'])
    print(res.shape)
    res = df.dropna(1,thresh=df.shape[0]-15)
    print(res.head())

    res = df.loc[df[['Height', 'Weight']].notna().all(1)]
    print(res.shape)
    res = df.loc[:,~(df.isna().sum()>15)]
    print(res.head())

# test1()

def test2():
    s = pd.Series([np.nan, 1, np.nan, np.nan, 2, np.nan],list('aaabcd'))
    print(s)
    print(s.fillna(method='ffill'))
    print(s.fillna(method='ffill',limit=1))
    print(s.fillna(s.mean()))
    print(s.fillna({'a':100,'d':200}))

    df = pd.read_csv('./docs/python/course3/data/learn_pandas.csv',
                     usecols = ['Grade', 'Name', 'Gender', 'Height',
                                'Weight', 'Transfer'])
    df_1 = df.groupby('Grade')['Height'].transform(lambda x:x.fillna(x.mean()))
    print(df_1.head())


# test2()

def test3():
    s = pd.Series([np.nan, np.nan, 1,np.nan, np.nan, np.nan,2, np.nan, np.nan])
    print(s)
    res = s.interpolate(limit_direction='backward',limit=1)
    print(res)
    res = s.interpolate(limit_direction='both',limit=1)
    print(res)
    print(s.interpolate("nearest").values)

    s = pd.Series([0,np.nan,10],index=[0,1,10])
    print(s)
    print(s.interpolate())
    print(s.interpolate(method='index'))
    s = pd.Series([0,np.nan,10],
                  index=pd.to_datetime(['20200101',
                                        '20200102',
                                        '20200111']))

    print(s)
    print(s.interpolate())
    print(s.interpolate(method='index'))


# test3()

def test4():
    # print(pd.to_timedelta(['30s',np.nan]))
    # print(pd.to_datetime(['20200101',np.nan]))
    df = pd.read_csv('./docs/python/course3/data/missing_chi.csv')
    print(df.head())
    print(df.isna().mean())
    print(df.y.value_counts(normalize=True))
    cat_1 = df.X_1.fillna('NaN').mask(df.X_1.notna()).fillna("NotNaN")
    cat_2 = df.X_2.fillna('NaN').mask(df.X_2.notna()).fillna("NotNaN")

    print(cat_1)
    print(cat_2)

    df_1 = pd.crosstab(cat_1,df.y,margins=True)
    df_2 = pd.crosstab(cat_2,df.y,margins=True)

    print(df_1)
    print(df_2)


test4()
