import scrapy
from deal.items import DayItem
import pymysql
from deal import settings
from deal.bll.CompanyInfoBll import CompanyInfoBll
from pandas import Series, DataFrame
import pandas as pd
import sys

class DaySpider(scrapy.Spider):
    name = "day"
    allowed_domains = ["hq.sinajs.cn"]

    company_list = None
    
    def start_requests(self):
        _EveryNum = 50
        _CompanyInfoBll=CompanyInfoBll()
        companys = _CompanyInfoBll.GetAllCompanyInfo()
        company_col_name=['id','code','market_code']
        self.company_list = DataFrame(list(companys), columns=company_col_name)

        urls_pre = 'http://hq.sinajs.cn/list='
        urls = ''
        for i in range(len(companys)):
            company = companys[i]
            if((i==0 or i%_EveryNum!=0) and  i+1!=len(companys)):
                urls = urls + str(company[2]) + str(company[1]) + ','
                continue
            request_url = urls_pre + urls + str(company[2]) + str(company[1])
            urls = ''
            yield scrapy.Request(url=request_url, callback=self.parse) 

    def parse(self, response):
        result = response.xpath('//html/body/p/text()').extract()
        allCompany_day = str(result).split(';')
        for company_day in allCompany_day:
            if(company_day.find('hq_str_')<0):
                continue
            record_list = str(company_day).split('"')
            cid_list = record_list[0].split('_')[2]
            cid = cid_list[2:8]
            info = record_list[1]
            info_list = info.split(',')
            dayItem = DayItem()
            dayItem['CompanyId'] = int(self.company_list[self.company_list['code']==cid].iat[0,0]) 
            dayItem['RDate'] = info_list[30]
            dayItem['open'] = float(info_list[1])
            dayItem['high'] = float(info_list[4])
            dayItem['close'] = float(info_list[3])
            dayItem['low'] = float(info_list[5])
            dayItem['volume'] = round(int(info_list[8])/100)
            dayItem['volume_price'] = round(float(info_list[9])/10000,2)
            yield dayItem
        