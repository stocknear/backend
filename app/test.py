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




identifier = 'AA250321C00045000'
next_page = ''

response = intrinio.OptionsApi().get_options_prices_eod(identifier, next_page=next_page, )
print(response)