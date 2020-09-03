# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class DealItem(scrapy.Item):
    code = scrapy.Field()
    name = scrapy.Field()
    market=scrapy.Field()
    pass

class DayItem(scrapy.Item):
    CompanyId = scrapy.Field()
    RDate = scrapy.Field()
    open = scrapy.Field()
    high = scrapy.Field()
    close = scrapy.Field()
    low = scrapy.Field()
    volume = scrapy.Field()
    volume_price = scrapy.Field()

    pass


class MMItem(scrapy.Item):
    CompanyId = scrapy.Field()
    RDateTime = scrapy.Field()
    current = scrapy.Field()
    volume = scrapy.Field()
    volume_price = scrapy.Field()
    m1 = scrapy.Field()
    m1_price = scrapy.Field()
    m2 = scrapy.Field()
    m2_price = scrapy.Field()
    m3 = scrapy.Field()
    m3_price = scrapy.Field()
    m4 = scrapy.Field()
    m4_price = scrapy.Field()
    m5 = scrapy.Field()
    m5_price = scrapy.Field()
    s1 = scrapy.Field()
    s1_price = scrapy.Field()
    s2 = scrapy.Field()
    s2_price = scrapy.Field()
    s3 = scrapy.Field()
    s3_price = scrapy.Field()
    s4 = scrapy.Field()
    s4_price = scrapy.Field()
    s5 = scrapy.Field()
    s5_price = scrapy.Field()

    pass
