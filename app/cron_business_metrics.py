from edgar import *

# Tell the SEC who you are
set_identity("Michael Mccallum mike.mccalum@indigo.com")


filings = Company("NVDA").get_filings(form="10-Q").latest(3)

print(filings.search("Revenue by Geography"))