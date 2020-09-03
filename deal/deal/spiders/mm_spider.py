import scrapy
from deal.items import MMItem
import pymysql
from deal import settings
from deal.bll.CompanyInfoBll import CompanyInfoBll
from pandas import Series, DataFrame
import pandas as pd
import sys

class DaySpider(scrapy.Spider):
    name = "mm"
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
            mmItem = MMItem()
            mmItem['CompanyId'] = int(self.company_list[self.company_list['code']==cid].iat[0,0]) 
            mmItem['RDateTime'] = str(info_list[30]) + ' ' + str(info_list[31])
            mmItem['current'] = float(info_list[3])
            mmItem['volume'] = round(int(info_list[8])/100)
            mmItem['volume_price'] = round(float(info_list[9])/10000,2)

            mmItem['m1'] = round(int(info_list[10])/100)
            mmItem['m1_price'] = round(float(info_list[11]))
            mmItem['m2'] = round(int(info_list[12])/100)
            mmItem['m2_price'] = round(float(info_list[13]))
            mmItem['m3'] = round(int(info_list[14])/100)
            mmItem['m3_price'] = round(float(info_list[15]))
            mmItem['m4'] = round(int(info_list[16])/100)
            mmItem['m4_price'] = round(float(info_list[17]))
            mmItem['m5'] = round(int(info_list[18])/100)
            mmItem['m5_price'] = round(float(info_list[19]))

            mmItem['s1'] = round(int(info_list[20])/100)
            mmItem['s1_price'] = round(float(info_list[21]))
            mmItem['s2'] = round(int(info_list[22])/100)
            mmItem['s2_price'] = round(float(info_list[23]))
            mmItem['s3'] = round(int(info_list[24])/100)
            mmItem['s3_price'] = round(float(info_list[25]))
            mmItem['s4'] = round(int(info_list[26])/100)
            mmItem['s4_price'] = round(float(info_list[27]))
            mmItem['s5'] = round(int(info_list[28])/100)
            mmItem['s5_price'] = round(float(info_list[29]))

            yield mmItem
        