import ujson

def save_json(data):
    with open(f"json/info-text/data.json", 'w') as file:
        ujson.dump(data, file)



data = {
    'researchDevelopmentRevenueRatio': {
        'text': 'The percentage of the company`s revenue that is spent on research and development.',
    },
    'avgVolume': {
        'text': 'The average daily volume over the last 20 trading days.',
    },
    'volume': {
        'text': 'The number of shares traded during the current or latest trading day.',
    },
    'rsi': {
        'text': 'The Relative Strength Index (RSI) measures whether a stock is overbought or oversold.',
    },
    'stochRSI': {
        'text': 'Stochastic RSI identifies overbought or oversold conditions in a stock.',
    },
    'mfi': {
        'text': 'The Money Flow Index (MFI) measures buying and selling pressure using price and volume.',
    },
    'cci': {
        'text': 'The Commodity Channel Index (CCI) identifies overbought or oversold conditions.',
    },
    'atr': {
        'text': 'The Average True Range (ATR) measures the volatility of an asset.',
    },
    'sma20': {
        'text': 'The average closing stock price over the last 20 days.',
    },
    'sma50': {
        'text': 'The average closing stock price over the last 50 days.',
    },
    'sma100': {
        'text': 'The average closing stock price over the last 100 days.',
    },
    'sma200': {
        'text': 'The average closing stock price over the last 200 days.',
    },
    'ema20': {
        'text': 'The exponentially weighted average of the stock closing price over the last 20 days.',
    },
    'ema50': {
        'text': 'The exponentially weighted average of the stock closing price over the last 50 days.',
    },
    'ema100': {
        'text': 'The exponentially weighted average of the stock closing price over the last 100 days.',
    },
    'ema200': {
        'text': 'The exponentially weighted average of the stock closing price over the last 200 days.',
    },
    'price': {
        'text': 'The current price of a single share.',
    },
    'change1W': {
        'text': 'The percentage change in the stock price compared to 1 week ago.',
    },
    'change1M': {
        'text': 'The percentage change in the stock price compared to 1 month ago.',
    },
    'change3M': {
        'text': 'The percentage change in the stock price compared to 3 months ago.',
    },
    'change6M': {
        'text': 'The percentage change in the stock price compared to 6 months ago.',
    },
    'change1Y': {
        'text': 'The percentage change in the stock price compared to 1 year ago.',
    },
    'change3Y': {
        'text': 'The percentage change in the stock price compared to 3 years ago.',
    },
    'marketCap': {
        'text': 'Market capitalization is the total value of all of a company`s outstanding shares.',
        'equation': 'Market Cap = Shares Outstanding * Stock Price'
    },
    'country': {
        'text': 'The country where the company has its primary headquarters.'
    },
    'revenue': {
        'text': 'Revenue is the amount of money a company receives from its main business activities, such as sales of products or services.'
    },
    'growthRevenue': {
        'text': 'Year-over-year (YoY) revenue growth is how much a company`s revenue has increased compared to the same time period one year ago.',
        'equation': 'Revenue Growth = ((Current Revenue / Previous Revenue) - 1) * 100%'
    },
    'costOfRevenue': {
        'text': 'Cost of revenue refers to the direct costs attributable to the production of the goods or services sold by a company, including materials, labor, and manufacturing overhead.'
    },
    'growthCostOfRevenue': {
        'text': 'Year-over-year (YoY) cost of revenue growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'costAndExpenses': {
        'text': 'Cost and expenses represent the total expenditures a company incurs to operate its business, including both direct costs associated with production and indirect costs such as administrative, marketing, and other operational expenses.',
    },
    'growthCostAndExpenses': {
        'text': 'Year-over-year (YoY) cost & expenses growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'netIncome': {
        'text': 'Net income is a company`s accounting profits after subtracting all costs and expenses from the revenue. It is also called earnings/profits',
        'equation': 'Net Income = Revenue - All Expenses'
    },
    'growthNetIncome': {
        'text': 'Year-over-year (YoY) net income growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'grossProfit': {
        'text': 'Gross profit is a company’s profit after subtracting the costs directly linked to making and delivering its products and services.',
        'equation': 'Gross Profit = Revenue - Cost of Revenue'
    },
    'growthGrossProfit': {
        'text': 'Year-over-year (YoY) gross profit growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'researchAndDevelopmentExpenses': {
        'text': 'Research and development (R&D) is an operating expense. It is the amount of money a company spends on researching and developing new products and services, or improving existing ones.',
    },
    'growthResearchAndDevelopmentExpenses': {
        'text': 'Year-over-year (YoY) R&D growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'payoutRatio': {
        'text': 'The dividend payout ratio is the percentage of a company`s profits that are paid out as dividends. A high ratio implies that the dividend payments may not be sustainable.',
        'equation': 'Payout Ratio = (Dividends Per Share / Earnings Per Share) * 100%'
    },
    'dividendYield': {
        'text': 'The dividend yield is how much a stock pays in dividends each year, as a percentage of the stock price.',
        'equation': 'Dividend Yield = (Annual Dividends Per Share / Stock Price) * 100%'
    },
    'annualDividend': {
        'text': 'The amount dividend that the company pays in total per year.',
    },
    'dividendGrowth': {
        'text': 'Year-over-year (YoY) dividend growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Payout Ratio = (Dividends Per Share / Earnings Per Share) * 100%'
    },
    'eps': {
        'text': 'Earnings per share is the portion of a company`s profit that is allocated to each individual stock. EPS is calculated by dividing net income by shares outstanding.',
        'equation': 'EPS = Net Income / Shares Outstanding'
    },
    'growthEPS': {
        'text': 'Year-over-year (YoY) eps growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'interestIncome': {
        'text': 'Interest Income is the revenue earned from interest payments on investments or loans held by a company. It is a key component of a company\'s overall income and reflects the return on invested funds.',
    },
    'interestExpenses': {
        'text': 'Interest Expenses is the cost incurred by a company for borrowed funds. It represents the amount paid to creditors for interest on outstanding debt and is an important factor in assessing a company\'s financial health and profitability.',
    },
    'growthInterestExpense': {
        'text': 'Year-over-year (YoY) interest expense growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'operatingExpenses': {
        'text': 'Operating Expenses are the costs associated with running a company’s core business operations. These expenses include rent, utilities, salaries, and other expenses necessary for maintaining daily operations, but exclude costs related to financing or investing activities.',
        'equation': 'Operating Expenses = Total Expenses - Non-Operating Expenses'
    },
    'growthOperatingExpenses': {
        'text': 'Year-over-year (YoY) operating expenses growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'operatingIncome': {
        'text': 'Operating Income represents the profit a company earns from its core business operations, excluding any income from non-operating activities such as investments or interest. It reflects the efficiency of the company’s core business activities and is a key indicator of operational performance.',
        'equation': 'Operating Income = Gross Profit - Operating Expenses'
    },
    'growthOperatingIncome': {
        'text': 'Year-over-year (YoY) operating income growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'freeCashFlow': {
        'text': 'Free cash flow is the cash remaining after the company spends on everything required to maintain and grow the business. It is calculated by subtracting capital expenditures from operating cash flow.',
        'equation': 'Free Cash Flow = Operating Cash Flow - Capital Expenditures'
    },
    'growthFreeCashFlow': {
        'text': 'Year-over-year (YoY) free cash flow growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'operatingCashFlow': {
        'text': 'Operating cash flow, also called cash flow from operating activities, measures the amount of cash that a company generates from normal business activities. It is the amount of cash left after all cash income has been received, and all cash expenses have been paid.',
    },
    'growthOperatingCashFlow': {
        'text': 'Year-over-year (YoY) operating cash flow growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'stockBasedCompensation': {
        'text': 'Stock-based compensation is the value of stocks issued for the purpose of compensating the executives and employees of a company.',
    },
    'growthStockBasedCompensation': {
        'text': 'Year-over-year (YoY) stock-based compensation growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'totalLiabilities': {
        'text': 'Total liabilities are all financial obligations of the company, including both current and long-term (non-current) liabilities. Liabilities are everything that the company owes.',
        'equation': 'Total Liabilities = Current Liabilities + Long-Term Liabilities'
    },
    'growthTotalLiabilities': {
        'text': 'Year-over-year (YoY) total liabilities growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'totalDebt': {
        'text': 'Total debt is the total amount of liabilities categorized as "debt" on the balance sheet. It includes both current and long-term (non-current) debt.',
        'equation': 'Total Debt = Current Debt + Long-Term Debt'
    },
    'growthTotalDebt': {
        'text': 'Year-over-year (YoY) total debt growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'totalStockholdersEquity': {
        'text': 'Shareholders’ equity is also called book value or net worth. It can be seen as the amount of money held by investors inside the company. It is calculated by subtracting all liabilities from all assets.',
        'equation': 'Shareholders Equity = Total Assets - Total Liabilities'
    },
    'growthTotalStockholdersEquity': {
        'text': 'Year-over-year (YoY) shareholders equity growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'cagr3YearRevenue': {
        'text': 'The compound annual growth rate of revenue over 3 years. CAGR is a calculation method often used to measure annual investment performance.',
        'equation': 'CAGR = ( (Final Value / Initial Value)^(1 / Number of Years) ) - 1'
    },
    'cagr5YearRevenue': {
        'text': 'The compound annual growth rate of revenue over 5 years. CAGR is a calculation method often used to measure annual investment performance.',
        'equation': 'CAGR = ( (Final Value / Initial Value)^(1 / Number of Years) ) - 1'
    },
    'cagr3YearEPS': {
        'text': 'The compound annual growth rate of eps over 3 years. CAGR is a calculation method often used to measure annual investment performance.',
        'equation': 'CAGR = ( (Final Value / Initial Value)^(1 / Number of Years) ) - 1'
    },
    'cagr5YearEPS': {
        'text': 'The compound annual growth rate of eps over 5 years. CAGR is a calculation method often used to measure annual investment performance.',
        'equation': 'CAGR = ( (Final Value / Initial Value)^(1 / Number of Years) ) - 1'
    },
    'returnOnInvestedCapital': {
        'text': 'Return on invested capital (ROIC) measures how effective a company is at investing its capital in order to increase profits. It is calculated by dividing the NOPAT (Net Operating Profit After Tax) by the invested capital.',
        'equation': 'ROIC = (NOPAT / (Debt + Equity)) * 100%'
    },
    'relativeVolume': {
        'text': 'The relative daily volume is the current day`s trading volume compared to the stocks 30-day average trading volume.',
    },
    'institutionalOwnership': {
        'text': 'Institutional Ownership Percentage indicates the proportion of a company’s shares that are owned by institutional investors, such as mutual funds, hedge funds and insurance companies. A higher percentage suggests strong institutional confidence in the company.',
    },
    'pe': {
        'text': 'The price-to-earnings (P/E) ratio is a valuation metric that shows how expensive a stock is relative to earnings.',
        'equation': 'PE Ratio = Stock Price / Earnings Per Share'
    },
    'forwardPE': {
        'text': 'The forward price-to-earnings (P/E) ratio is like the PE ratio, except that it uses the estimated earnings over the next year instead of historical earnings.',
        'equation': 'Forward PE = Stock Price / Forward EPS (1Y)'
    },
    'forwardPS': {
        'text': 'The forward price-to-sales (P/S) ratio is like the PS ratio, except that it uses next year`s forecasted revenue instead of the revenue over the last 12 months.',
        'equation': 'Forward PS = Market Capitalization / Revenue Next Year'
    },
    'priceToBookRatio': {
        'text': 'The price-to-book (P/B) ratio measures a stock`s price relative to book value. Book value is also called Shareholders equity.',
        'equation': 'PB Ratio = Market Capitalization / Shareholders Equity'
    },
    'priceToSalesRatio': {
        'text': 'The price-to-sales (P/S) ratio is a commonly used valuation metric. It shows how expensive a stock is compared to revenue.',
        'equation': 'PS Ratio = Market Capitalization / Revenue'
    },
    'beta': {
        'text': 'Beta measures the price volatility of a stock in comparison to the overall stock market. A value higher than 1 indicates greater volatility, while a value under 1 indicates less volatility.',
    },
    'ebitda': {
        'text': 'EBITDA stands for "Earnings Before Interest, Taxes, Depreciation and Amortization." It is a commonly used measure of profitability.',
        'equation': 'EBITDA = Net Income + Interest + Taxes + Depreciation and Amortization'
    },
    'growthEBITDA': {
        'text': 'Year-over-year (YoY) ebitda growth is how much a company has increased compared to the same time period one year ago.',
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
    },
    'var': {
        'text': 'Value at Risk (VaR) measures the maximum potential loss an investment portfolio could face over a specified time period, given a certain confidence level. It provides a statistical estimate of the worst-case scenario under normal market conditions.',
    },

}


save_json(data)

