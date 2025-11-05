import scrapy
import json


class ZilspiderSpider(scrapy.Spider):
    name = "zilspider"
    allowed_domains = ["zillow.com"]
    url = "https://www.zillow.com/philadelphia-pa/rentals/"

    def start_requests(self):
        yield scrapy.Request(url=self.url, callback=self.parse)
    
    def parse(self, response):

        next_data = response.xpath("//script[@id='__NEXT_DATA__']/text()").get()
        homes = next_data['props']['pageProps']['searchPageState']['cat1']['searchResults']['listResults']
        pass