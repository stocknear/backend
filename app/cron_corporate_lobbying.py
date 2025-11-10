# Imports from python.
from csv import DictWriter
from datetime import datetime
import json
from math import ceil
import os
from time import sleep
import pandas as pd
from dotenv import load_dotenv
import requests
from collections import defaultdict
import math
from fuzzywuzzy import process
import sqlite3
import concurrent.futures

BASE_SESSION = requests.Session()

BASE_API_URL = "https://lda.senate.gov/api/v1"

load_dotenv()
API_KEY = os.getenv('SENATE_API_KEY')


LDA_API_ENDPOINTS = dict(
    filing_types=f"{BASE_API_URL}/constants/filing/filingtypes/",
    filings=f"{BASE_API_URL}/filings/",
)

# Sadly, the Senate lowered the max results per page to 25.
# RESULTS_PER_PAGE = 250
RESULTS_PER_PAGE = 250

TIME_PERIOD_SLUGS = dict(
    Q1="first_quarter",
    Q2="second_quarter",
    Q3="third_quarter",
    Q4="fourth_quarter",
    MY="mid_year",
    YE="year_end",
)

TIME_PERIOD_PREFIXES = dict(
    Q1="1st Quarter",
    Q2="2nd Quarter",
    Q3="3rd Quarter",
    Q4="4th Quarter",
    MY="Mid-Year",
    YE="Year-End",
)


def parse_safe_query_dict(raw_dict):
    return "&".join([f"{k}={v}" for k, v in raw_dict.items()])

def querystring_to_dict(raw_url):
    return {
        k: v
        for d in [
            dict([_.split("=")]) for _ in raw_url.split("?")[1].split("&")
        ]
        for k, v in d.items()
    }


with open("json/corporate-lobbying/self_lobbying_overrides.json", "r") as input:
    SELF_LOBBYING_OVERRIDES = json.load(input)


def get_types_for_quarter(time_period, common_session=None):
    session = BASE_SESSION if common_session is None else common_session

    rq = requests.Request(
        "GET",
        LDA_API_ENDPOINTS["filing_types"],
        headers={
            "Accept-Encoding": "gzip,deflate,br",
            "Accept": "application/json",
            "Authorization": f'Token {API_KEY}',
        },
    ).prepare()

    request_result = session.send(rq)

    if 200 <= request_result.status_code <= 299:
        all_types = json.loads(request_result.text)

        return [
            type_dict
            for type_dict in all_types
            if type_dict["name"].startswith(TIME_PERIOD_PREFIXES[time_period])
        ]

    return []


def get_filings_page(time_config, common_session=None, extra_fetch_params={}):
    session = BASE_SESSION if common_session is None else common_session

    query_dict = dict(
        **time_config,
        # ordering="-dt_posted,id",
        ordering="dt_posted,id",
        page_size=RESULTS_PER_PAGE,
        **extra_fetch_params,
    )

    rq = requests.Request(
        "GET",
        f"{LDA_API_ENDPOINTS['filings']}?{parse_safe_query_dict(query_dict)}",
        headers={
            "Accept-Encoding": "gzip,deflate,br",
            "Accept": "application/json",
            "Authorization": f'Token {API_KEY}',
        },
    ).prepare()

    request_result = session.send(rq)

    if 200 <= request_result.status_code <= 299:
        return dict(
            range=200,
            status=request_result.status_code,
            headers=request_result.headers,
            body=json.loads(request_result.text),
        )
    elif 400 <= request_result.status_code <= 499:
        return dict(
            range=400,
            status=request_result.status_code,
            headers=request_result.headers,
            body=None,
        )

    return dict(
        status=request_result.status_code,
        headers=request_result.headers,
        body=None,
    )


def commonize(raw_value):
    formatted_value = (
        raw_value.lower()
        .replace(".", "")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace(" u s a ", " usa ")
        .replace(" u.s. ", " us ")
        .replace(" u s ", " us ")
        .replace("  ", " ")
        .strip()
    )

    if formatted_value.endswith(" us a"):
        formatted_value = f"{formatted_value[:-5]} usa"
    elif formatted_value.endswith(" u s"):
        formatted_value = f"{formatted_value[:-4]} us"

    if formatted_value.startswith("the "):
        return formatted_value[4:]

    elif formatted_value.startswith("u.s. "):
        return f"us {formatted_value[5:]}"
    elif formatted_value.startswith("u s "):
        return f"us {formatted_value[4:]}"

    return formatted_value


def process_result(raw_result, type_dict):
    posting_date = datetime.fromisoformat(raw_result["dt_posted"])

    registrant_name = raw_result["registrant"]["name"]
    client_name = raw_result["client"]["name"]

    amount_reported = raw_result["income"]
    amount_type = "income"

    if all(
        [
            raw_result["income"] is None,
            commonize(registrant_name) == commonize(client_name),
        ]
    ):
        amount_reported = raw_result["expenses"]
        amount_type = "expenses"

    if amount_type == "income" and raw_result["income"] is None:
        matching_overrides = [
            override_dict
            for override_dict in SELF_LOBBYING_OVERRIDES
            if override_dict["registrantName"] == registrant_name
            and override_dict["clientName"] == client_name
        ]

        if matching_overrides:
            amount_reported = raw_result["expenses"]
            amount_type = "expenses*"

    return dict(
        UUID=raw_result["filing_uuid"],
        RegistrantName=registrant_name,
        ClientName=client_name,
        FilingType=type_dict[raw_result["filing_type"]].replace(" - ", " "),
        AmountReported=amount_reported,
        DatePosted=posting_date.strftime("%Y-%m-%d"),
        FilingYear=raw_result["filing_year"],
        AmountType=amount_type,
    )


def collect_filings(time_config, type_dict, session):
    current_page = get_filings_page(time_config, session)

    results_count = current_page["body"]["count"]
    results_lang = "filings" if results_count != 1 else "filing"

    page_count = ceil(results_count / RESULTS_PER_PAGE)
    page_lang = "pages" if page_count != 1 else "page"

    print(f"  ### {results_count} {results_lang} / {page_count} {page_lang}")

    all_filings = [
        process_result(result, type_dict)
        for result in current_page["body"]["results"]
    ]

    print("  - PAGE 1")

    while current_page["body"]["next"] is not None:
        next_query_dict = querystring_to_dict(current_page["body"]["next"])

        next_query_diff = {
            k: v
            for k, v in next_query_dict.items()
            if k not in [*time_config.keys(), "ordering", "page_size"]
        }

        sleep(1)

        current_page = get_filings_page(time_config, session, next_query_diff)

        print(f"  - PAGE {next_query_diff['page']}")

        all_filings.extend(
            [
                process_result(result, type_dict)
                for result in current_page["body"]["results"]
            ]
        )

    return all_filings


def scrape_lda_filings(year, time_period, common_session=None):
    session = BASE_SESSION if common_session is None else common_session

    types_for_period = get_types_for_quarter(time_period, session)

    type_dict = {
        filing_type["value"]: filing_type["name"]
        for filing_type in types_for_period
    }

    all_filings = {}

    for filing_type in types_for_period:
        print("")

        print(f"{filing_type['name']} ({filing_type['value']}):")

        time_config = dict(
            filing_year=year,
            filing_period=TIME_PERIOD_SLUGS[time_period],
            filing_type=filing_type["value"],
        )

        all_filings[filing_type["value"]] = collect_filings(
            time_config,
            type_dict,
            session,
        )

        print("")

    with open(f"json/corporate-lobbying/reports/{year}-{time_period.lower()}.csv", "w") as output_file:
        writer = DictWriter(
            output_file,
            fieldnames=[
                "UUID",
                "RegistrantName",
                "ClientName",
                "FilingType",
                "AmountReported",
                "DatePosted",
                "FilingYear",
                "AmountType",
            ],
        )
        writer.writeheader()

        for type_slug, filings_for_type in all_filings.items():
            for filing in filings_for_type:
                writer.writerow(filing)

    return all_filings


def get_historical_data():
    current_year = datetime.now().year
    year_list = list(range(2015, current_year + 1))
    quarter_list = ['Q1', 'Q2', 'Q3', 'Q4']
    print(year_list)
    
    for year in year_list:
        for quarter in quarter_list:
            file_name = f"{year}-{quarter.lower()}"
            print(file_name)
            if not os.path.exists(f"json/corporate-lobbying/reports/{file_name}.csv"):
                scrape_lda_filings(year, quarter)
            else:
                print(f"Skipping {file_name}, file already exists.")

def get_current_quarter_and_year():
    current_date = datetime.now()
    current_month = current_date.month
    current_year = str(current_date.year)
    
    if 1 <= current_month <= 3:
        quarter = "Q1"
    elif 4 <= current_month <= 6:
        quarter = "Q2"
    elif 7 <= current_month <= 9:
        quarter = "Q3"
    else:
        quarter = "Q4"
    
    return current_year, quarter

def update_latest_quarter():
    year, quarter = get_current_quarter_and_year()
    print(year, quarter)
    scrape_lda_filings(year, quarter)

def save_json(symbol, data):
    with open(f"json/corporate-lobbying/companies/{symbol}.json", 'w') as file:
        json.dump(data, file)


def process_stocks_batch(stocks, csv_files, reports_folder, threshold):
    all_df = pd.concat([pd.read_csv(os.path.join(reports_folder, csv_file), usecols=['ClientName', 'AmountReported', 'FilingYear']) for csv_file in csv_files])
    all_df['ClientName_lower'] = all_df['ClientName'].str.lower()
    
    results = {}
    for stock in stocks:
        print(stock['name'])
        stock_name_lower = stock['name'].lower()
        
        all_df['score'] = all_df['ClientName_lower'].apply(lambda x: process.extractOne(stock_name_lower, [x])[1])
        matched_df = all_df[all_df['score'] >= threshold]
        
        year_totals = matched_df.groupby('FilingYear')['AmountReported'].sum().to_dict()
        all_res_list = [{'year': year, 'amount': amount} for year, amount in year_totals.items()]
        
        if all_res_list:
            save_json(stock['symbol'], all_res_list)
            print(f"Saved data for {stock['symbol']} ({len(all_res_list)} matches)")
        
        results[stock['symbol']] = all_res_list
    
    return results

def create_dataset():
    reports_folder = "json/corporate-lobbying/reports"
    threshold = 95
    csv_files = [f for f in os.listdir(reports_folder) if f.endswith('.csv')]
    
    con = sqlite3.connect('stocks.db')
    cursor = con.cursor()
    cursor.execute("PRAGMA journal_mode = wal")
    cursor.execute("SELECT DISTINCT symbol, name FROM stocks WHERE symbol NOT LIKE '%.%' AND symbol NOT LIKE '%-%'")
    stock_data = [{'symbol': row[0], 'name': row[1]} for row in cursor.fetchall()]
    print(f"Total stocks: {len(stock_data)}")
    con.close()
    
    batch_size = 3
    stock_batches = [stock_data[i:i+batch_size] for i in range(0, len(stock_data), batch_size)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_stocks_batch, batch, csv_files, reports_folder, threshold) for batch in stock_batches]
        
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            print(f"Processed batch with {len(results)} stocks")



if '__main__' == __name__:

    #get_historical_data()
    update_latest_quarter()
    create_dataset()
