import numpy as np
import pandas as pd

def test1():
    ts = pd.Timestamp('2020/1/1')
    print(ts)
    ts = pd.Timestamp('2020-1-1 08:10:30')
    print(ts)
    print(ts.year,ts.month,ts.day,ts.hour,ts.minute,ts.second)
    print(pd.Timestamp.max)
    print(pd.Timestamp.min)
    print(pd.to_datetime(['2020-1-1', '2020-1-3', '2020-1-6']))
    df = pd.read_csv('./docs/python/course3/data/learn_pandas.csv')
    print(pd.to_datetime(df.Test_Date).head())
    temp = pd.to_datetime(['2020\\1\\1','2020\\1\\3'],format='%Y\\%m\\%d')
    print(temp)
    df_date_cols = pd.DataFrame({'year': [2020, 2020],
                                 'month': [1, 1],
                                 'day': [1, 2],
                                 'hour': [10, 20],
                                 'minute': [30, 50],
                                 'second': [20, 40]})
    print(pd.to_datetime(df_date_cols))
    print(pd.date_range('2020-1-1','2020-1-21', freq='10D'))
    print(pd.date_range('2020-1-1','2020-2-28', freq='10D'))
    print(pd.date_range('2020-1-1','2020-2-28', periods=6))
    s = pd.Series(np.random.rand(5),
                index=pd.to_datetime([
                    '2020-1-%d'%i for i in range(1,10,2)]))

    print(s)
    print(s.asfreq('D'))
    print(s.asfreq('12H'))


# test1()

def test2():
    s = pd.Series(pd.date_range('2020-1-1','2020-1-3', freq='D'))
    print(s.dt.date)
    print(s.dt.time)
    print(s.dt.day)
    print(s.dt.daysinmonth)
    print(s.dt.dayofweek)
    print(s.dt.month_name())
    print(s.dt.day_name())
    print(s.dt.is_year_start)
    print(s.dt.is_year_end)
    s = pd.Series(pd.date_range('2020-1-1 20:35:00','2020-1-1 22:35:00',freq='45min'))
    print(s)
    print(s.dt.round('1H'))
    print(s.dt.ceil('1H'))
    print(s.dt.floor('1H'))
    s = pd.Series(np.random.randint(2,size=366),index=pd.date_range('2020-01-01','2020-12-31'))
    print(s.head())
    idx = pd.Series(s.index).dt
    print(s[(idx.is_month_start | idx.is_month_end).values].head())
    print(s[idx.dayofweek.isin([5,6]).values].head())
    print(s['2020-01-01'])
    print(s['20200101'])
    print(s['2020-07'].head())
    print(s['2020-05':'2020-7-15'].head())
    print(s['2020-05':'2020-7-15'].tail())

# test2()

def test3():
    print(pd.Timestamp('20200102 08:00:00')-pd.Timestamp('20200101 07:35:00'))
    print(pd.Timedelta(days=1,minutes=25))
    print(pd.Timedelta('1 days 25 minutes'))

    df = pd.read_csv('./docs/python/course3/data/learn_pandas.csv')
    s = pd.to_timedelta(df.Time_Record)
    print(s.head())
    print(pd.timedelta_range('0s','1000s',freq='6min'))
    print(pd.timedelta_range('0s','1000s',periods=3))
    print(s.dt.seconds.head())
    print(s.dt.total_seconds().head())
    print(s.dt.round('min').head())

    td1 = pd.Timedelta(days=1)
    td2 = pd.Timedelta(days=3)
    ts = pd.Timestamp('20200101')
    print(td1 * 2)
    print(td2 - td1)
    print(ts + td1)
    print(ts - td1)

    td1 = pd.timedelta_range(start='1 days', periods=5)
    td2 = pd.timedelta_range(start='12 hours',freq='2H',periods=5)
    ts = pd.date_range('20200101', '20200105')
    print(td1 * 5)
    print(td1 * pd.Series(list(range(5))))
    print(td1 - td2)
    print(td1 + pd.Timestamp('20200101'))
    print(td1 + ts)

    print(pd.Timestamp('20200831') + pd.offsets.WeekOfMonth(week=0,weekday=0))
    print(pd.Timestamp('20200907') + pd.offsets.BDay(30))
    print(pd.Timestamp('20200831') - pd.offsets.WeekOfMonth(week=0,weekday=0))
    print(pd.Timestamp('20200907') - pd.offsets.BDay(30))
    print(pd.Timestamp('20200907') + pd.offsets.MonthEnd())

    my_filter = pd.offsets.CDay(n=1,weekmask='Wed Fri',holidays=['20200109'])
    dr = pd.date_range('20200108', '20200111')
    print(dr.to_series().dt.dayofweek)
    print([i + my_filter for i in dr])

# test3()

import matplotlib.pyplot as plt

def test4():
    print(pd.date_range('20200101','20200331', freq='MS'))
    print(pd.date_range('20200101','20200331', freq='M'))
    print(pd.date_range('20200101','20200110', freq='B'))
    print(pd.date_range('20200101','20200201', freq='W-MON'))
    print(pd.date_range('20200101','20200201',freq='WOM-1MON'))

    print(pd.date_range('20200101','20200331',freq=pd.offsets.MonthBegin()))
    print(pd.date_range('20200101','20200331',freq=pd.offsets.MonthEnd()))
    print(pd.date_range('20200101','20200110', freq=pd.offsets.BDay()))
    print(pd.date_range('20200101','20200201',freq=pd.offsets.CDay(weekmask='Mon')))
    print(pd.date_range('20200101','20200201',freq=pd.offsets.WeekOfMonth(week=0,weekday=0)))

    idx = pd.date_range('20200101', '20201231', freq='B')
    np.random.seed(2020)
    data = np.random.randint(-1,2,len(idx)).cumsum()
    s = pd.Series(data,index=idx)
    print(s.head())

    # r = s.rolling('30D')
    # plt.plot(s)
    # plt.title('BOLL LINES')
    # plt.plot(r.mean())
    # plt.plot(r.mean() + r.std() *2)
    # plt.plot(r.mean() - r.std()*2)
    # plt.show()

    print(s.shift(freq='50D').head())
    print(s.resample('10D').mean().head())
    print(s.resample('10D').apply(lambda x:x.max()-x.min()).head())

    idx = pd.date_range('20200101 8:26:35', '20200101 9:31:58', freq='77s')
    data = np.random.randint(-1,2,len(idx)).cumsum()
    s = pd.Series(data,index=idx)
    print(s.head())
    print(s.resample('7min').mean().head())
    print(s.resample('7min', origin='start').mean().head())

    s = pd.Series(np.random.randint(2,size=366),index=pd.date_range('2020-01-01','2020-12-31'))
    print(s.resample('M').mean().head())
    print(s.resample('MS').mean().head())


# test4()




# 将 Datetime, Time 合并为一个时间列 Datetime ，同时把它作为索引后排序。
# 每条记录时间的间隔显然并不一致，请解决如下问题：
# 找出间隔时间的前三个最大值所对应的三组时间戳。
# 是否存在一个大致的范围，使得绝大多数的间隔时间都落在这个区间中？如果存在，请对此范围内的样本间隔秒数画出柱状图，设置 bins=50 。
# 求如下指标对应的 Series ：
# 温度与辐射量的6小时滑动相关系数
# 以三点、九点、十五点、二十一点为分割，该观测所在时间区间的温度均值序列
# 每个观测6小时前的辐射量（一般而言不会恰好取到，此时取最近时间戳对应的辐射量）

def test5():
    df = pd.read_csv('./docs/python/course3/data/solar.csv', usecols=['Data','Time','Radiation','Temperature'])
    print(df.head())
    solar_date = pd.to_datetime(df.Data).dt.date.astype('string')
    df['Data'] = pd.to_datetime(solar_date + " " + df.Time)
    df = df.drop(columns='Time').rename(columns={'Data':'Datetime'}).set_index('Datetime').sort_index()
    print(df.head())
    s = df.index.to_series().reset_index(drop=True).diff().dt.total_seconds()
    max_3 = s.nlargest(3).index
    print(df.index[max_3.union(max_3-1)])

    # res = s.mask((s>s.quantile(0.99))|(s<s.quantile(0.01)))
    # plt.hist(res, bins=50)
    # plt.show()

    res = df.Radiation.rolling('6H').corr(df.Temperature)
    print(res.tail(3))
    res = df.Temperature.resample('6H', origin='03:00:00').mean()
    print(res.head(3))

    # print(df.index)
    my_dt = df.index.shift(freq='-6H')
    # print(my_dt)
    int_loc = [df.index.get_loc(i, method='nearest') for i in my_dt]
    res = df.Radiation.iloc[int_loc]
    print(res.tail(3))




# test5()





    # 统计如下指标：
    # 每月上半月（15号及之前）与下半月葡萄销量的比值
    # 每月最后一天的生梨销量总和
    # 每月最后一天工作日的生梨销量总和
    # 每月最后五天的苹果销量均值
    # 按月计算周一至周日各品种水果的平均记录条数，行索引外层为水果名称，内层为月份，列索引为星期。
    # 按天计算向前10个工作日窗口的苹果销量均值序列，非工作日的值用上一个工作日的结果填充。

def test6():
    df = pd.read_csv('./docs/python/course3/data/fruit.csv')
    print(df.head())

test6()

