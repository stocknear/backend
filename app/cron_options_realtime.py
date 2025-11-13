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

identifier = 'GME'

response = intrinio.OptionsApi().get_options_greeks_by_ticker(identifier)
print(response)
    
