# Scrapy settings for zillow_scraper project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "zillow_scraper"

SPIDER_MODULES = ["zillow_scraper.spiders"]
NEWSPIDER_MODULE = "zillow_scraper.spiders"

ADDONS = {}


# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = "zillow_scraper (+http://www.yourdomain.com)"

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Concurrency and throttling settings
CONCURRENT_REQUESTS = 1
#CONCURRENT_REQUESTS_PER_DOMAIN = 1
DOWNLOAD_DELAY = 4

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
# "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
# "Accept-Language": "en",

DEFAULT_REQUEST_HEADERS = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9,el;q=0.8',
    'cache-control': 'max-age=0',
    'priority': 'u=0, i',
    'referer': 'https://www.zillow.com/professionals/real-estate-agent-reviews/philadelphia-pa/',
    'sec-ch-ua': '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
    # 'cookie': 'zguid=24|%24dbcace1c-94cd-45b0-b327-ed65ba09bd4d; zgsession=1|8a98ec24-e637-446f-a5b2-566c2a9ba73e; pxcts=ce527f4d-9dbd-11f0-831c-8b57c4086623; _pxvid=ce5278da-9dbd-11f0-831b-db14663a6b75; _ga=GA1.2.1449374766.1759209897; zjs_anonymous_id=%22dbcace1c-94cd-45b0-b327-ed65ba09bd4d%22; zjs_user_id=null; zg_anonymous_id=%2225ca5697-9d2c-4a26-ac51-d2a3892c21f2%22; zjs_user_id_type=%22encoded_zuid%22; _gcl_au=1.1.311437620.1759209899; datagrail_consent_id=7e84c9ce-057e-4c91-87ef-56e6d4914637.2e0a09e1-4d9c-4a66-8268-1067fcfabcc7; datagrail_consent_id_s=7e84c9ce-057e-4c91-87ef-56e6d4914637.c83f1183-481c-4251-9d4f-bc5f3d2a7fdd; _scid=PZKbOazb1BViZirhOQOEc1anV7pCLe35; _tt_enable_cookie=1; _ttp=01K6CH9JH4KHNS64XCY8B25TRM_.tt.1; DoubleClickSession=true; _pin_unauth=dWlkPVpEWm1OV1F3WXpJdE4yVTJNaTAwTkRaakxXRm1NR0l0Tnpaa056bGpPR1V6TkRjMA; _fbp=fb.1.1759209901085.802511451559682523; _lr_env_src_ats=false; _gid=GA1.2.262660381.1759957025; _sctr=1%7C1759896000000; _ScCbts=%5B%5D; _clck=ujl9jf%5E2%5Eg00%5E0%5E2099; JSESSIONID=AA15E25AFE9B9A6F504461A0B73FC1BF; tfpsi=cf8bbeec-07e1-4fee-8028-54ba58252d9c; _lr_retry_request=true; FSsampler=1640628656; web-platform-data=%7B%22wp-dd-rum-session%22%3A%7B%22doNotTrack%22%3Atrue%7D%7D; _rdt_uuid=1759209899593.e2b9f95e-e38e-4c65-950b-7ccab1abb726; _scid_r=SRKbOazb1BViZirhOQOEc1anV7pCLe359PuRtQ; __gads=ID=0ed63acf3b820da1:T=1759209914:RT=1759976638:S=ALNI_MbkzJao7-naKrCEM5Ezb6W2xTRNiQ; __gpi=UID=00001160153e29b8:T=1759209914:RT=1759976638:S=ALNI_MZsvgUiFPQbwwI5z1XwYw4NZpSujw; __eoi=ID=d9144672b7abed17:T=1759209914:RT=1759976638:S=AA-AfjZjcO31a--sFgsQarG98Ofi; _dd_s=isExpired=1; AWSALB=in4Q1FGa2i6YpC94UQ0W2fUOnQ7olA+HOK7RZCpa04EAMfxs4OLQ3CqiE3Vbo8KEQuLUwav1aif6DLJeVndrQUFwcJkRhIuVTG8jCvMxANTKZSRntgOL3ZZ19Q7r; AWSALBCORS=in4Q1FGa2i6YpC94UQ0W2fUOnQ7olA+HOK7RZCpa04EAMfxs4OLQ3CqiE3Vbo8KEQuLUwav1aif6DLJeVndrQUFwcJkRhIuVTG8jCvMxANTKZSRntgOL3ZZ19Q7r; _px3=b26b690ce5695f3ccdabd82ce3ed2396d4cac232026977a6bf36b10a50697b70:q5MpoB9CW2wBNeHTXbK574zht/h5Oib7AhcKNeGet/HXFhjJiEIJ8tLtqwO8HnQE7l42t+hr4t5E1//OoFIRqg==:1000:a+63ZyfO5ZE2uUk5/kWJhQlENVYOWlR8Ac3A9M+x8+mGSkbHzbScA6gHaog70yA8QKuiWigtfNsabsEk1bJq4owA6CNMvsn1iqD8fLxPBmducmh1oPBWpoIuv3vBSUDZBMa7HaAm6GMfExeh08+0HTys6a3uJJk/bO/mukGx5SA7iv5KVAducRBtw2Xuvxcu6NVUTvPPI2Ybxg11k2HopakSp6Q/w1EP8XqVfHrN8vs=; search=6|1762568669411%7Crect%3D40.23722206534691%2C-74.70257936718752%2C39.74904764175662%2C-75.54852663281252%26rid%3D13271%26disp%3Dmap%26mdm%3Dauto%26p%3D1%26listPriceActive%3D1%26type%3Dhouse%2Ccondo%2Ctownhouse%2Capartment%26fs%3D0%26fr%3D1%26mmm%3D0%26rs%3D0%26singlestory%3D0%26housing-connector%3D0%26parking-spots%3Dnull-%26abo%3D0%26garage%3D0%26pool%3D0%26ac%3D0%26waterfront%3D0%26finished%3D0%26unfinished%3D0%26cityview%3D0%26mountainview%3D0%26parkview%3D0%26waterview%3D0%26hoadata%3D1%26zillow-owned%3D0%263dhome%3D0%26showcase%3D0%26featuredMultiFamilyBuilding%3D0%26onlyRentalStudentHousingType%3D0%26onlyRentalIncomeRestrictedHousingType%3D0%26onlyRentalMilitaryHousingType%3D0%26onlyRentalDisabledHousingType%3D0%26onlyRentalSeniorHousingType%3D0%26excludeNullAvailabilityDates%3D0%26isRoomForRent%3D0%26isEntirePlaceForRent%3D1%26ita%3D0%26stl%3D0%26fur%3D0%26os%3D0%26ca%3D0%26np%3D0%26hasDisabledAccess%3D0%26hasHardwoodFloor%3D0%26areUtilitiesIncluded%3D0%26highSpeedInternetAvailable%3D0%26elevatorAccessAvailable%3D0%26commuteMode%3Ddriving%26commuteTimeOfDay%3Dnow%09%0913271%09%7B%22isList%22%3Atrue%2C%22isMap%22%3Afalse%7D%09%09%09%09%09; _uetsid=5b31fed0a48911f09ca15d00dc29eff2; _uetvid=cfb0b8609dbd11f087b87fc8f0e94d5c; _clsk=1xy8k0f%5E1759976667257%5E17%5E0%5Es.clarity.ms%2Fcollect; connectId=%7B%22vmuid%22%3A%22zVyG9qDbXUjqtruwFboMSV4Jf9XZp_rPuTVTLxW9Z4ZyDoXZqzo0Gsw9zPrIQaksdmmoqK9FUzZp1igN1VrxDg%22%2C%22connectid%22%3A%22zVyG9qDbXUjqtruwFboMSV4Jf9XZp_rPuTVTLxW9Z4ZyDoXZqzo0Gsw9zPrIQaksdmmoqK9FUzZp1igN1VrxDg%22%2C%22connectId%22%3A%22zVyG9qDbXUjqtruwFboMSV4Jf9XZp_rPuTVTLxW9Z4ZyDoXZqzo0Gsw9zPrIQaksdmmoqK9FUzZp1igN1VrxDg%22%2C%22ttl%22%3A86400000%2C%22lastSynced%22%3A1759957049313%2C%22lastUsed%22%3A1759976667526%7D; ttcsid=1759974576010::bbRwm6h5z5qSZjWK6Lrk.6.1759976667703.0; ttcsid_CN5P33RC77UF9CBTPH9G=1759974576010::TMXgo7BM1Jobys7Ppzvn.6.1759976667703.0',
}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    "zillow_scraper.middlewares.ZillowScraperSpiderMiddleware": 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    "zillow_scraper.middlewares.ZillowScraperDownloaderMiddleware": 543,
#}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    "scrapy.extensions.telnet.TelnetConsole": None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
#ITEM_PIPELINES = {
#    "zillow_scraper.pipelines.ZillowScraperPipeline": 300,
#}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = "httpcache"
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Set settings whose default value is deprecated to a future-proof value
FEED_EXPORT_ENCODING = "utf-8"
