from __future__ import print_function
import time
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('INTRINIO_API_KEY')



intrinio.ApiClient().set_api_key(api_key)
intrinio.ApiClient().allow_retries(True)




source = 'delayed'
start_date = ''
start_time = ''
end_date = ''
end_time = ''
timezone = 'UTC'
page_size = 100
min_size = 100
security = 'AAPL'
next_page = ''

response = intrinio.OptionsApi().get_option_trades(source=source, start_date=start_date, start_time=start_time, end_date=end_date, end_time=end_time, timezone=timezone, page_size=page_size, min_size=min_size, security=security, next_page=next_page)
print(response)
    