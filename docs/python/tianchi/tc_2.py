import pandas as pd

# 读取候选人信息，由于原始数据没有表头，需要添加表头
candidates = pd.read_csv("./docs/python/tianchi/data/weball20.txt", sep = '|',names=['CAND_ID','CAND_NAME','CAND_ICI','PTY_CD','CAND_PTY_AFFILIATION','TTL_RECEIPTS',
                                                          'TRANS_FROM_AUTH','TTL_DISB','TRANS_TO_AUTH','COH_BOP','COH_COP','CAND_CONTRIB',
                                                          'CAND_LOANS','OTHER_LOANS','CAND_LOAN_REPAY','OTHER_LOAN_REPAY','DEBTS_OWED_BY',
                                                          'TTL_INDIV_CONTRIB','CAND_OFFICE_ST','CAND_OFFICE_DISTRICT','SPEC_ELECTION','PRIM_ELECTION','RUN_ELECTION'
                                                          ,'GEN_ELECTION','GEN_ELECTION_PRECENT','OTHER_POL_CMTE_CONTRIB','POL_PTY_CONTRIB',
                                                          'CVG_END_DT','INDIV_REFUNDS','CMTE_REFUNDS'])


# print(candidates.head())

# 读取候选人和委员会的联系信息
ccl = pd.read_csv("./docs/python/tianchi/data/ccl.txt", sep = '|',names=['CAND_ID','CAND_ELECTION_YR','FEC_ELECTION_YR','CMTE_ID','CMTE_TP','CMTE_DSGN','LINKAGE_ID'])
# print(ccl.head())

# 关联两个表数据
ccl = pd.merge(ccl,candidates)
# print(ccl.head())
# 提取出所需要的列
ccl = pd.DataFrame(ccl, columns=[ 'CMTE_ID','CAND_ID', 'CAND_NAME','CAND_PTY_AFFILIATION'])
# print(ccl.head())

    # CMTE_ID：委员会ID
    # CAND_ID：候选人ID
    # CAND_NAME：候选人姓名
    # CAND_PTY_AFFILIATION：候选人党派



# 读取个人捐赠数据，由于原始数据没有表头，需要添加表头
# 提示：读取本文件大概需要5-10s
itcont = pd.read_csv('./docs/python/tianchi/data/itcont_2020_20200722_20200820.txt', sep='|',names=['CMTE_ID','AMNDT_IND','RPT_TP','TRANSACTION_PGI',
                                                                                  'IMAGE_NUM','TRANSACTION_TP','ENTITY_TP','NAME','CITY',
                                                                                  'STATE','ZIP_CODE','EMPLOYER','OCCUPATION','TRANSACTION_DT',
                                                                                  'TRANSACTION_AMT','OTHER_ID','TRAN_ID','FILE_NUM','MEMO_CD',
                                                                                  'MEMO_TEXT','SUB_ID'])
# print(itcont.head())

# 将候选人与委员会关系表ccl和个人捐赠数据表itcont合并，通过 CMTE_ID
c_itcont =  pd.merge(ccl,itcont)
# 提取需要的数据列
c_itcont = pd.DataFrame(c_itcont, columns=[ 'CAND_NAME','NAME', 'STATE','EMPLOYER','OCCUPATION',
                                           'TRANSACTION_AMT', 'TRANSACTION_DT','CAND_PTY_AFFILIATION'])


    # CAND_NAME – 接受捐赠的候选人姓名
    # NAME – 捐赠人姓名
    # STATE – 捐赠人所在州
    # EMPLOYER – 捐赠人所在公司
    # OCCUPATION – 捐赠人职业
    # TRANSACTION_AMT – 捐赠数额（美元）
    # TRANSACTION_DT – 收到捐款的日期
    # CAND_PTY_AFFILIATION – 候选人党派

# print(c_itcont.head())
# print(c_itcont.shape)
# print(c_itcont.info())

c_itcont['STATE'].fillna('NOT PROVIDED',inplace=True)
c_itcont['EMPLOYER'].fillna('NOT PROVIDED',inplace=True)
c_itcont['OCCUPATION'].fillna('NOT PROVIDED',inplace=True)

c_itcont['TRANSACTION_DT'] = c_itcont['TRANSACTION_DT'].astype(str)
c_itcont['TRANSACTION_DT'] = [i[3:] + i[:3] for i in c_itcont['TRANSACTION_DT']]  

# print(c_itcont.head())
# print(c_itcont.info())
# print(c_itcont.describe())
# print(c_itcont['CAND_NAME'].describe())

# print(c_itcont.groupby('CAND_PTY_AFFILIATION').sum().sort_values('TRANSACTION_AMT',ascending=False).head(10))
# print(c_itcont.groupby('CAND_NAME').sum().sort_values('TRANSACTION_AMT',ascending=False).head(10))
# print(c_itcont.groupby('OCCUPATION').sum().sort_values('TRANSACTION_AMT',ascending=False).head(10))
# print(c_itcont['OCCUPATION'].value_counts().head(10))
# print(c_itcont.groupby('STATE').sum().sort_values('TRANSACTION_AMT',ascending=False).head(10))
# print(c_itcont['STATE'].value_counts().head(10))

import matplotlib.pyplot as plt
from wordcloud import WordCloud,ImageColorGenerator

# st_amt = c_itcont.groupby('STATE').sum().sort_values('TRANSACTION_AMT',ascending=False)[:10]
# st_amt = pd.DataFrame(st_amt,columns=['TRANSACTION_AMT'])
# st_amt.plot(kind='bar')
# plt.show()

# st_amt = c_itcont.groupby('STATE').size().sort_values(ascending=False)[:10]
# st_amt.plot(kind='bar')
# plt.show()

biden = c_itcont[c_itcont['CAND_NAME']=='BIDEN, JOSEPH R JR']
# biden_state = biden.groupby('STATE').sum().sort_values('TRANSACTION_AMT',ascending=False)[:10]
# biden_state.plot.pie(figsize=(10,10),autopct='%0.2f%%',subplots=True)
# plt.show()

# 在4.2 热门候选人拜登在各州的获得的捐赠占比 中我们已经取出了所有支持拜登的人的数据，存在变量：biden中
# 将所有捐赠者姓名连接成一个字符串
# data = ' '.join(biden["NAME"].tolist())
# # 读取图片文件
# bg = plt.imread("./docs/python/tianchi/data/biden.jpg")
# # 生成
# wc = WordCloud(# FFFAE3
#     background_color="white",  # 设置背景为白色，默认为黑色
#     width=890,  # 设置图片的宽度
#     height=600,  # 设置图片的高度
#     mask=bg,    # 画布
#     margin=10,  # 设置图片的边缘
#     max_font_size=100,  # 显示的最大的字体大小
#     random_state=20,  # 为每个单词返回一个PIL颜色
# ).generate_from_text(data)

# # 图片背景
# bg_color = ImageColorGenerator(bg)
# # 开始画图
# plt.imshow(wc.recolor(color_func=bg_color))
# # 为云图去掉坐标轴
# plt.axis("off")
# # 画云图，显示
# # 保存云图
# wc.to_file("biden_wordcloud.png")


# 按州总捐款热力地图
'''
参赛选手自由发挥、补充
第一个补充热力地图的参赛选手可以获得天池杯子一个
'''

# 收到捐赠额最多的两位候选人的总捐赠额变化趋势
'''
参赛选手自由发挥、补充
第一个补充捐赠额变化趋势图的参赛选手可以获得天池杯子一个
'''

# 其他可视化方向
'''
参赛选手自由发挥、补充
官方将选取5个创新可视化的选手，送出天池杯子一个
'''