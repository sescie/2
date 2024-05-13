import scrapy
from scrapy_splash import SplashRequest
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError
from twisted.internet.error import TimeoutError, TCPTimedOutError

class Spider1(scrapy.Spider):
    name = "spider1"
    start_urls = [
        "https://www.thezimbabwemail.com/category/politics/",
        "https://www.thezimbabwemail.com/category/business/",
        "https://www.thezimbabwemail.com/category/sports/",
        "https://www.thezimbabwemail.com/category/entertainment/",
    ]

    def parse(self, response):
            # Extract category from the updated structure

            for article in response.css('.SectionModule_module__GAZLm'):
                category = response.css('h1.page-title::text').get()

                title = article.css('.entry-title a::text').get()
                link = article.css('.entry-title a::attr(href)').get()
                content = article.css('.mh-excerpt p::text').get()

                scraped_info = {
                    'category': category,
                    'title': title,
                    'link': link,
                    'content': content
                }
                yield scraped_info


class Spider2(scrapy.Spider):
    name = "spider2"
    start_urls = [
        'https://www.newsday.co.zw/category/4/business',
        'https://www.newsday.co.zw/category/9/opinion-and-analysis',
        'https://www.newsday.co.zw/category/5/sport',
        'https://www.newsday.co.zw/category/8/lifestyle-and-arts'
    ]

    def parse(self, response):
        # Extract category from the updated structure

            category = response.css('div.brand-title h2 a.links::text').get()
            title = response.css('a.text-dark div.sub-title.mt-3::text').extract()
            link = response.css('div.card-body.pad-o.mt-3 a::attr(href)').extract()
            content = response.css('div.card-body.pad-o.mt-3 div.mb-3.pt-2.top-article::text').extract()
            data = zip(title,link,content)
            for row in data:

                scraped_info = {
                    'category': category,
                    'title': row[0],
                    'link': row[1],
                    'content': row[2]
                }
                yield scraped_info
class Spider3(scrapy.Spider):
    name = "spider3"
    start_urls = [
        "https://apnews.com/politics",
        "https://apnews.com/sports",
        "https://apnews.com/business",
        "https://apnews.com/entertainment",
    ]

    def parse(self, response):
        category = response.css('bsp-custom-headline h1.PageList-header-title::text').get()
        for article in response.css('.PagePromo-content'):
            title = article.css('bsp-custom-headline > h3.PagePromo-title a::text').get(default='')
            link = article.css('h3.PagePromo-title > a::attr(href)').get(default='')
            content = article.css('div.PagePromo-description > a span::text').get(default='')

            scraped_info = {
            'category': category,
            'title': title,
            'link': link,
            'content': content
            }
            yield scraped_info

class Spider4(scrapy.Spider):
    name = "spider4"
    start_urls = [
        "https://www.afrogazette.co.zw/category/tie-business/",
        # "https://www.afrogazette.co.zw/category/crime-courts/",
        # "https://www.afrogazette.co.zw/category/sports/",
        # "https://www.afrogazette.co.zw/category/bizarre/",
    ]

    def parse(self, response):          

                category = response.css('h1.page-title::text').get()
                for article in response.css('.post-details'):
                    title = article.css('h2.post-title a::text').get()
                    link = article.css('h2.post-title a::attr(href)').get()
                    content = article.css('.post-excerpt p::text').get()

                    scraped_info = {
                        'category': category,
                        'title': title,
                        'link': link,
                        'content': content
                    }
                    yield scraped_info