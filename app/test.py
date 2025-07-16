from datetime import datetime, date, timedelta
from collections import defaultdict
from tqdm import tqdm
import sqlite3
import orjson
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import math


with open("json/options-flow/feed/data.json","rb") as file:
    data = orjson.loads(file.read())
    data = sorted(data, key=lambda x: x['cost_basis'], reverse=True)
    print(data[0])