
# 尝试执行以下代码，并解释错误原因
# class C:
#     def myFun():
#         print('Hello!')

# class C:
#     def myFun(self):
#         print('Hello!')

# c = C()
# c.myFun()


# 按照以下要求定义一个游乐园门票的类，并尝试计算2个成人+1个小孩平日票价。
# 要求:
# 1. 平日票价100元
# 2. 周末票价为平日的120%
# 3. 儿童票半价


class Ticket():

    def __init__(self):
        self.adult_price = 100
        self.child_price = 50
        self.weekend_rate = 1.2

    def get_price(self,adult_num,child_num,is_weekend):
        if is_weekend:
            return (self.adult_price * adult_num + self.child_price * child_num)* self.weekend_rate
        else:
            return self.adult_price * adult_num + self.child_price * child_num
        

# ticket = Ticket()
# print(ticket.get_price(2,1,False))
# print(ticket.get_price(2,1,True))



# 利用python做一个简单的定时器类
# 要求:
# a. 定制一个计时器的类。
# b. start 和 stop 方法代表启动计时和停止计时。
# c. 假设计时器对象 t1 ， print(t1) 和直接调用 t1 均显示结果。
# d. 当计时器未启动或已经停止计时时，调用 stop 方法会给予温馨的提示。
# e. 两个计时器对象可以进行相加： t1+t2 。
# f. 只能使用提供的有限资源完成




# 了解Collection模块，编写程序以查询给定列表中最常见的元素。
# 题目说明:
# 输入: language = ['PHP', 'PHP', 'Python', 'PHP', 'Python', 'JS', 'Python', 'Python','PHP', 'Python']
# 输出: Python

import collections

# print(help(collections))

def most_element():
    language = ['PHP', 'PHP', 'Python', 'PHP', 'Python', 'JS', 'Python', 'Python','PHP', 'Python']
    max_mapping = collections.Counter(language).most_common(1)
    
    print(max_mapping[0][0])

# most_element()


# 1. 距离你出生那天过去多少天了？
# 2. 距离你今年的下一个生日还有多少天？
# 3. 将距离你今年的下一个生日的天数转换为秒数。

import datetime

# today = datetime.date.today()
# todaytime = datetime.datetime(today.year,today.month,today.day)
# birthday = datetime.datetime(2000,1,3)
# print(today.strftime("%Y/%m/%d %H:%M:%S"))
# print(birthday)

# dt = todaytime - birthday
# print(dt.days)

# nextbirdaytime = datetime.datetime(today.year+1,1,3)
# dt2 = nextbirdaytime - todaytime
# print(dt2.days)
# print(dt2.total_seconds())
# print(dt2.days*24*60*60)



# 假设你获取了用户输入的日期和时间如 2020-1-21 9:01:30 ，以及一个时区信息如 UTC+5:00 ，均是 str ，请编写
# 一个函数将其转换为timestamp：


# Input file
# example1: dt_str='2020-6-1 08:10:30', tz_str='UTC+7:00'
# example2: dt_str='2020-5-31 16:10:30', tz_str='UTC-09:00'
# Output file
# result1: 1590973830.0
# result2: 1590973830.0

from dateutil import parser 

def to_timestamp(dt_str, tz_str):
    if "+" in tz_str:
        dt = parser.parse(dt_str + "+" + tz_str.split("+")[1])
    elif "-" in tz_str:
        dt = parser.parse(dt_str + "-" + tz_str.split("-")[1])
    else:
        dt = parser.parse(dt_str+ tz_str)
    
    print(dt.timestamp())
    

# to_timestamp(dt_str='2020-6-1 08:10:30', tz_str='UTC+7:00')
# to_timestamp(dt_str='2020-5-31 16:10:30', tz_str='UTC-09:00')


# 2. 编写Python程序以选择指定年份的所有星期日。

def all_sunday(year):
    dt_start = datetime.datetime(year,1,1)
    dt_sunday_one = datetime.datetime(year,1,1 + 6 - dt_start.weekday())
    dt_weekday = datetime.timedelta(days=7)
    dt_end = datetime.datetime(year,12,31)
    print(dt_sunday_one.strftime("%Y-%m-%d"))
    while True:
        dt_sunday_one = dt_sunday_one + dt_weekday
        if dt_sunday_one > dt_end:
            break
        print(dt_sunday_one.strftime("%Y-%m-%d"))
    pass

# all_sunday(2019)


import os

# print(os.getcwd())
# print(os.name)
# os.system("calc")


# 编写程序查找最长的单词
def test_file():
    file_path = "./docs/python/course3/data/1.txt"
    word_list = []
    f = open(file_path,'r',encoding="UTF-8")
    lines = f.readlines()
    for line in lines:
        for word in line.split():
            word_list.append(word)

    f.close()
    
    max_len_word = word_list[0]
    max_len_word_list = []
    max_len_word_list.append(max_len_word)

    for word in word_list:
        if len(word) > len(max_len_word):
            max_len_word = word
            max_len_word_list.clear()
            max_len_word_list.append(max_len_word)
        elif len(word) == len(max_len_word):
            max_len_word_list.append(word)

    print(max_len_word_list)

test_file()
