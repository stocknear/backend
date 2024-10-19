from edgar import *
import ast

# Tell the SEC who you are
set_identity("Michael Mccallum mike.mccalum@indigo.com")

symbol = 'AAPL'
filings = Company(symbol).get_filings(form="10-Q").latest(10)
#print(filings[0].xbrl())

filing_xbrl = filings[0].xbrl()

# ----
# Extract detailed revenue items
# ----
revenue_sources = []
facts = filing_xbrl.facts.data

latest_rows = facts.groupby('dimensions').head(1)

for index, row in latest_rows.iterrows():
    dimensions_str = row.get("dimensions", "{}")
    try:
        dimensions_dict = ast.literal_eval(dimensions_str) if isinstance(dimensions_str, str) else dimensions_str
    except (ValueError, SyntaxError):
        dimensions_dict = {}
                
    product_dimension = dimensions_dict.get("srt:ProductOrServiceAxis") if isinstance(dimensions_dict, dict) else None


    if row["namespace"] == "us-gaap" and row["fact"].startswith("Revenue") and product_dimension is not None and product_dimension.startswith(symbol.lower() + ":") :
        revenue_sources.append({
            "name": product_dimension.replace("Member", "").replace(f"{symbol.lower()}:", ""),
            "value": row["value"],
        })

print(revenue_sources)
