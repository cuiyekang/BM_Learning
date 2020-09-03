# -*- coding: utf-8 -*-


import pymysql
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import sys

conn = pymysql.connect(host='192.168.209.170', user='root', passwd='Cyk@821007', db='DealSystem', charset='utf8')
cur = conn.cursor()


sql =  'select a.RId,a.CompanyCode,b.MarketCode from CompanyInfo a inner join Market b on a.MarketId=b.RId LIMIT 0,10'
cur.execute(sql)
companys = cur.fetchall() 

cur.close()
conn.close()

company_col_name=['id','code','market_code']
#company_list = DataFrame(list(companys), columns=company_col_name)
urls_pre = 'http://hq.sinajs.cn/list={0}{1}'

urls = ''
for i in range(len(companys)):
    company = companys[i]
    if((i==0 or i%4!=0) and  i+1!=len(companys)):
        urls = urls + str(company[2]) + str(company[1]) + ','
        continue
    request_url = urls_pre + urls + str(company[2]) + str(company[1])
    print(request_url)
    urls = ''

#for company in companys:
#    request_url = urls_pre.format(company[2],company[1])
#    cid = request_url.split("=")[1]
#   cid=cid[2:]
#    print(cid)
