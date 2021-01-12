import numpy as np
import pandas as pd

def test1():
    var = 'abcd'
    print(str.upper(var))
    s = pd.Series(['abcd','efg','hi'])
    print(s.str.upper())
    print(var[0])
    print(var[-1:0:-2])
    print(s.str[0])
    print(s.str[-1:0:-2])
    print(s.str[2])
    s = pd.Series([{1: 'temp_1', 2: 'temp_2'}, ['a', 'b'], 0.5, 'my_string'])
    print(s)
    print(s.str[1])
    print(s.astype('string').str[1])

# test1()

import re

def test2():
    print(re.findall(r'Apple','Apple! This Is an Apple!'))
    print(re.findall(r'.','abc'))
    print(re.findall(r'[ac]','abc'))
    print(re.findall(r'[^ac]','abc'))
    print(re.findall(r'[ab]{2}','aaaabbbb'))
    print(re.findall(r'aaa|bbb','aaaabbbb'))
    print(re.findall(r'a\\?|a\*','aa?a*a'))
    print(re.findall(r'a?.','abaacadaae'))
    print(re.findall(r'.s','Apple! This Is an Apple!'))
    print(re.findall(r'\w{2}','09 8? 7w c_ 9q p@'))
    print(re.findall(r'\w\W\B','09 8? 7w c_ 9q p@'))
    print(re.findall(r'.\s.','Constant dropping wears the stone.'))
    print(re.findall(r'上海市(.{2,3}区)(.{2,3}路)(\d+号)','上海市黄浦区方浜中路249号 上海市宝山区密山路5号'))
    s = pd.Series(['上海市黄浦区方浜中路249号','上海市宝山区密山路5号'])
    print(s.str.split('[市区路]'))
    print(s.str.split('[市区路]',n=2,expand=True))
    print(s.str.rsplit('[市区路]', n=2, expand=True))


# test2()

def test3():
    s = pd.Series([['a','b'], [1, 'a'], [['a', 'b'], 'c']])
    print(s.str.join('-'))
    s1 = pd.Series(['a','b'])
    s2 = pd.Series(['cat','dog'])
    print(s1.str.cat(s2,sep='-'))
    s2.index = [1,2]
    print(s1.str.cat(s2,sep='-',na_rep='?',join='outer'))
    s = pd.Series(['my cat', 'he is fat', 'railway station'])
    print(s.str.contains('\s\wat'))
    print(s.str.startswith('my'))
    print(s.str.endswith('t'))
    print(s.str.match('m|h'))
    print(s.str[::-1].str.match('ta[f|g]|n'))
    print(s.str.contains('^[m|h]'))
    print(s.str.contains('[f|g]at|n$'))
    s = pd.Series(['This is an apple. That is not an apple.'])
    print(s.str.find('apple'))
    print(s.str.rfind('apple'))
    s = pd.Series(['a_1_b','c_?'])
    print(s.str.replace('\d|\?','new',regex=True))
    



# test3()


# s = pd.Series(['上海市黄浦区方浜中路249号','上海市宝山区密山路5号','北京市昌平区北农路2号'])
# pat = '(\w+市)(\w+区)(\w+路)(\d+号)'
# city = {'上海市': 'Shanghai', '北京市': 'Beijing'}
# district = {'昌平区': 'CP District','黄浦区': 'HP District','宝山区': 'BS District'}
# road = {'方浜中路': 'Mid Fangbin Road','密山路': 'Mishan Road','北农路': 'Beinong Road'}

# def my_func(m):
#     str_city = city[m.group(1)]
#     str_district = district[m.group(2)]
#     str_road = road[m.group(3)]
#     str_no = 'NO. ' + m.group(4)[:-1]
#     return ' '.join([str_city,str_district,str_road,str_no])

# print(s.str.replace(pat,my_func,regex=True))

# pat = '(?P<市名>\w+市)(?P<区名>\w+区)(?P<路名>\w+路)(?P<编号>\d+号)'

# def my_func1(m):
#     str_city = city[m.group('市名')]
#     str_district = district[m.group('区名')]
#     str_road = road[m.group('路名')]
#     str_no = 'NO. ' + m.group('编号')[:-1]
#     return ' '.join([str_city,str_district,str_road,str_no])

# print(s.str.replace(pat,my_func1,regex=True))

# pat = '(\w+市)(\w+区)(\w+路)(\d+号)'
# print(s.str.extract(pat))
# pat = '(?P<市名>\w+市)(?P<区名>\w+区)(?P<路名>\w+路)(?P<编号>\d+号)'
# print(s.str.extract(pat))

# s = pd.Series(['A135T15,A26S5','B674S2,B25T6'], index = ['my_A','my_B'])
# pat = '[A|B](\d+)[T|S](\d+)'
# print(s.str.extractall(pat))
# pat_with_name = '[A|B](?P<name1>\d+)[T|S](?P<name2>\d+)'
# print(s.str.extractall(pat_with_name))
# print(s.str.findall(pat))


# s = pd.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
# print(s.str.upper())
# print(s.str.lower())
# print(s.str.title())
# print(s.str.capitalize())
# print(s.str.swapcase())

# s = pd.Series(['1', '2.2', '2e', '??', '-2.1', '0'])
# print(pd.to_numeric(s,errors='ignore'))
# print(pd.to_numeric(s,errors='coerce'))
# print(s[pd.to_numeric(s,errors='coerce').isna()])

# s = pd.Series(['cat rat fat at', 'get feed sheet heat'])
# print(s.str.count('[r|f]at|ee'))
# print(s.str.len())

# my_index = pd.Index([' col1', 'col2 ', ' col3 '])
# print(my_index.str.strip().str.len())
# print(my_index.str.rstrip().str.len())
# print(my_index.str.lstrip().str.len())

# s = pd.Series(['a','b','c'])
# print(s.str.pad(5,'left','*'))
# print(s.str.pad(5,'right','*'))
# print(s.str.pad(5,'both','*'))
# print(s.str.rjust(5,'*'))
# print(s.str.ljust(5,'*'))
# print(s.str.center(5,'*'))

# s = pd.Series([7, 155, 303000]).astype('string')
# print(s.str.pad(6,'left','0'))
# print(s.str.rjust(6,'0'))
# print(s.str.zfill(6))




# 将 year 列改为整数年份存储。
# 将 floor 列替换为 Level, Highest 两列，其中的元素分别为 string 类型的层类别（高层、中层、低层）与整数类型的最高层数。
# 计算房屋每平米的均价 avg_price ，以 ***元/平米 的格式存储到表中，其中 *** 为整数。


df = pd.read_excel('./docs/python/course3/data/house_info.xls', usecols=['floor','year','area','price'])
print(df.head())
df.year = pd.to_numeric(df.year.str[:-2]).astype('Int64')
print(df.head())
pat = '(\w层)（共(\d+)层）'
new_cols = df.floor.str.extract(pat).rename(columns={0:'Level', 1:'Highest'})
df = pd.concat([df.drop(columns=['floor']), new_cols], 1)
print(df.head())
s_area = pd.to_numeric(df.area.str[:-1])
s_price = pd.to_numeric(df.price.str[:-1])
df['avg_price'] = ((s_price/s_area)*10000).astype('int').astype('string') + '元/平米'
print(df.head())

