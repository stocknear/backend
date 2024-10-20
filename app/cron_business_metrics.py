from edgar import *
import ast
from tqdm import tqdm
from datetime import datetime

# Define quarter-end dates for a given year
#The last quarter Q4 result is not shown in any sec files
#But using the https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm 10-K you see the annual end result which can be subtracted with all Quarter results to obtain Q4 (dumb af but works so don't judge me people)


def closest_quarter_end(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year = date.year
    
    # Define quarter end dates for the year
    q1 = datetime(year, 3, 31)
    q2 = datetime(year, 6, 30)
    q3 = datetime(year, 9, 30)
    q4 = datetime(year, 12, 31)
    
    # Find the closest quarter date
    closest = min([q1, q2, q3, q4], key=lambda d: abs(d - date))
    
    # Return the closest quarter date in 'YYYY-MM-DD' format
    return closest.strftime("%Y-%m-%d")
    
# Tell the SEC who you are
set_identity("Michael Mccallum mike.mccalum@indigo.com")

symbol = 'NVDA'
revenue_sources = []
geography_sources = []
filings = Company(symbol).get_filings(form=["10-K","10-Q"]).latest(50)
#print(filings[0].xbrl())

for i in range(0,17):
    try:
        filing_xbrl = filings[i].xbrl()
        facts = filing_xbrl.facts.data
        latest_rows = facts.groupby('dimensions').head(1)


        for index, row in latest_rows.iterrows():
            dimensions_str = row.get("dimensions", "{}")
            try:
                dimensions_dict = ast.literal_eval(dimensions_str) if isinstance(dimensions_str, str) else dimensions_str
            except (ValueError, SyntaxError):
                dimensions_dict = {}

            for column_name in ["srt:StatementGeographicalAxis","srt:ProductOrServiceAxis"]:

                product_dimension = dimensions_dict.get(column_name) if isinstance(dimensions_dict, dict) else None
                #print(product_dimension)
                #print(row["namespace"], row["fact"], product_dimension, row["value"])
                
                if column_name == "srt:ProductOrServiceAxis":
                    if row["namespace"] == "us-gaap" and product_dimension is not None and (product_dimension.startswith(symbol.lower() + ":") or product_dimension.startswith('country' + ":")):
                        revenue_sources.append({
                            "name": product_dimension.replace("Member", "").replace(f"{symbol.lower()}:", ""),
                            "value": row["value"], "date": row["end_date"]
                        })

                else:
                    if row["namespace"] == "us-gaap" and product_dimension is not None and (product_dimension.startswith(symbol.lower() + ":") or product_dimension.startswith('country' + ":")):
                        geography_sources.append({
                            "name": product_dimension.replace("Member", "").replace(f"{symbol.lower()}:", ""),
                            "value": row["value"], "date": row["end_date"]
                        })


    except Exception as e:
        print(e)


#print(revenue_sources)
print(geography_sources)
