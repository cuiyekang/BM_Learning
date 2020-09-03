# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import pymysql
from deal import settings
from deal.bll.CompanyInfoBll import CompanyInfoBll

class DealPipeline(object):

    _companyInfoBll=None

    def open_spider(self, spider):
        self._companyInfoBll=CompanyInfoBll() 

    def process_item(self, item, spider):
        if spider.name == 'deal':
            self._companyInfoBll.AddCompanyInfo(item)
        elif spider.name == 'day':
            self._companyInfoBll.AddCompanyDayInfo(item)
        elif spider.name == 'mm':
            self._companyInfoBll.AddCompanyMMInfo(item)
        return item