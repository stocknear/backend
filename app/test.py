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




#identifier = 'AA250321C00045000'

symbol = 'MSFT'
strike = 95
source = ''
stock_price_source = ''
model = ''
show_extended_price = ''
include_related_symbols = False

response = intrinio.OptionsApi().get_option_strikes_realtime(symbol, strike, source=source, stock_price_source=stock_price_source, model=model, show_extended_price=show_extended_price, include_related_symbols=include_related_symbols)
print(response)