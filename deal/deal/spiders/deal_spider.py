import scrapy
from deal.items import DealItem
from deal.bll.CompanyInfoBll import CompanyInfoBll
from pandas import Series, DataFrame
import pandas as pd

class DealSpider(scrapy.Spider):
    name = "deal"
    allowed_domains = ["app.finance.ifeng.com"]
    
    market_list=None

    def start_requests(self):
        _CompanyInfoBll=CompanyInfoBll()
        results=_CompanyInfoBll.GetAllMarket()
        col_name = ['id', 'name', 'code']
        self.market_list = DataFrame(list(results), columns=col_name)

        urls = [
        'http://app.finance.ifeng.com/list/stock.php?t=ha&f=symbol&o=asc&p=1',
        'http://app.finance.ifeng.com/list/stock.php?t=sa&f=symbol&o=asc&p=1'
        ]
        for request_url in urls:
            yield scrapy.Request(url=request_url, callback=self.parse) 
        
    def parse(self, response):
        links = response.xpath('//*[@class="tab01"]/table/tr[position()>1]')
        t=response.xpath('//div[@class="block"]/h1/text()')
        mName = t.extract()[0]
        print(mName)
        for link in links:
            code = link.xpath('td[1]/a/text()').extract()
            name = link.xpath('td[2]/a/text()').extract()
            dealItem = DealItem()
            dealItem['code'] = code
            dealItem['name'] = name
            dealItem['market'] = int(self.market_list[self.market_list['name']==mName].iat[0,0]) 
            yield dealItem
        for url in response.xpath('//*[@class= "tab01"]/table/tr[52]/td/a/@href').extract():
            url = "http://app.finance.ifeng.com/list/stock.php" + url
            yield scrapy.Request(url, callback = self.parse)
        