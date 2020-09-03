# -*- coding: utf-8 -*-

import pymysql
from deal import settings

class CompanyInfoBll:

    conn = None
    cur = None

    def __init__(self):
        self.conn = pymysql.connect(host=settings.MYSQL_HOST, user=settings.MYSQL_USER, passwd=settings.MYSQL_PASSWD, db=settings.MYSQL_DBNAME, charset='utf8')
        self.cur = self.conn.cursor()

    def AddCompanyInfo(self,item):
        insert_sql = """insert into CompanyInfo(CompanyCode,CompanyName,MarketId) VALUES(%s,%s,%s)  on duplicate key update CompanyCode=(CompanyCode)"""
        self.cur.execute(insert_sql,(item['code'],item['name'],item['market']))
        self.conn.commit()

    def GetAllMarket(self):
        sql =  'select * from Market'
        self.cur.execute(sql)
        results = self.cur.fetchall() 
        return results

    def GetAllCompanyInfo(self):
        sql =  'select a.RId,a.CompanyCode,b.MarketCode from CompanyInfo a inner join Market b on a.MarketId=b.RId'
        self.cur.execute(sql)
        results = self.cur.fetchall() 
        return results
    
    def AddCompanyDayInfo(self,item):
        insert_sql = """insert into CompanyDayInfo(CompanyId,RDate,open,high,close,low,price_change,p_change,volume,volume_price,ma5,ma10,ma20,v_ma5,v_ma10,v_ma20,turnover) 
                                                   VALUES(%s,%s,%s,%s,%s,%s,0,0,%s,%s,0,0,0,0,0,0,0)"""
        self.cur.execute(insert_sql,(item['CompanyId'],item['RDate'],item['open'],item['high'],item['close'],item['low'],item['volume'],item['volume_price']))
        self.conn.commit()
    
    def AddCompanyMMInfo(self,item):
        insert_sql = """insert into CompanyMMInfo(CompanyId,RDateTime,current,price_change,p_change,volume,volume_price,m1,m1_price,m2,m2_price,m3,m3_price,m4,m4_price,m5,m5_price,s1,s1_price,s2,s2_price,s3,s3_price,s4,s4_price,s5,s5_price) 
                                                   VALUES(%s,%s,%s,0,0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        self.cur.execute(insert_sql,(item['CompanyId'],item['RDateTime'],item['current'],item['volume'],item['volume_price'],item['m1'],item['m1_price'],item['m2'],item['m2_price'],item['m3'],item['m3_price'],item['m4'],item['m4_price'],item['m5'],item['m5_price'],item['s1'],item['s1_price'],item['s2'],item['s2_price'],item['s3'],item['s3_price'],item['s4'],item['s4_price'],item['s5'],item['s5_price']))
        self.conn.commit()

    def __del__(self):
        self.cur.close()
        self.conn.close()

    
