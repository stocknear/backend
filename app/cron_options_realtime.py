import threading
import signal
import time
import sys
import logging
from threading import Event, Lock

from intriniorealtime.options_client import IntrinioRealtimeOptionsClient
from intriniorealtime.options_client import OptionsQuote
from intriniorealtime.options_client import OptionsTrade
from intriniorealtime.options_client import OptionsRefresh
from intriniorealtime.options_client import OptionsUnusualActivity
from intriniorealtime.options_client import OptionsUnusualActivityType
from intriniorealtime.options_client import OptionsUnusualActivitySentiment
from intriniorealtime.options_client import log
from intriniorealtime.options_client import Config
from intriniorealtime.options_client import Providers
from intriniorealtime.options_client import LogLevel

import os
from dotenv import load_dotenv


options_refresh_count = 0
options_refresh_count_lock = Lock()

load_dotenv()

api_key = os.getenv('INTRINIO_API_KEY')


def on_refresh(refresh: OptionsRefresh):
    global options_refresh_count
    global options_refresh_count_lock
    with options_refresh_count_lock:
        options_refresh_count += 1

def on_trade(trade: OptionsTrade):
    print(OptionsTrade)


class Summarize(threading.Thread):
    def __init__(self, stop_flag: threading.Event, intrinio_client: IntrinioRealtimeOptionsClient):
        threading.Thread.__init__(self, group=None, args=(), kwargs={}, daemon=True)
        self.__stop_flag: threading.Event = stop_flag
        self.__client = intrinio_client

    def run(self):
        while not self.__stop_flag.is_set():
            time.sleep(30.0)
            (dataMsgs, txtMsgs, queueDepth) = self.__client.get_stats()
            print(self.__client.get_stats())



# Your config object MUST include the 'api_key' and 'provider', at a minimum
config: Config = Config(
    api_key=api_key,
    provider=Providers.OPTIONS_EDGE, # or Providers.OPTIONS_EDGE
    num_threads=8,
    symbols=["GME"], # this is a static list of symbols (options contracts or option chains) that will automatically be subscribed to when the client starts
    log_level=LogLevel.INFO,
    delayed=False) #set delayed parameter to true if you have realtime access but want the data delayed 15 minutes anyway

# Register only the callbacks that you want.
# Take special care when registering the 'on_quote' handler as it will increase throughput by ~10x
intrinioRealtimeOptionsClient: IntrinioRealtimeOptionsClient = IntrinioRealtimeOptionsClient(config, on_trade=on_trade, on_refresh=on_refresh)

stop_event = Event()


def on_kill_process(sig, frame):
    log("Sample Application - Stopping")
    stop_event.set()
    intrinioRealtimeOptionsClient.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, on_kill_process)

summarize_thread = Summarize(stop_event, intrinioRealtimeOptionsClient)
summarize_thread.start()

intrinioRealtimeOptionsClient.start()

#use this to join the channels already declared in your config
intrinioRealtimeOptionsClient.join()

time.sleep(60 * 60 * 24)
# sigint, or ctrl+c, during the thread wait will also perform the same below code.
on_kill_process(None, None)