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
        'equation': 'Value Growth = ((Current Value / Previous Value) - 1) * 100%'
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
        'equation': 'ROIC = (NOPAT / (Debt + Equity))'
    },
    'relativeVolume': {
        'text': 'The relative daily volume is the current day`s trading volume compared to the stocks 30-day average trading volume.',
    },
    'institutionalOwnership': {
        'text': 'Institutional Ownership Percentage indicates the proportion of a company’s shares that are owned by institutional investors, such as mutual funds, hedge funds and insurance companies. A higher percentage suggests strong institutional confidence in the company.',
    },
    'priceToEarningsRatio': {
        'text': 'The price-to-earnings (P/E) ratio is a valuation metric that shows how expensive a stock is relative to earnings.',
        'equation': 'PE Ratio = Stock Price / Earnings Per Share'
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
    'trendAnalysis': {
        'text': 'Accuracy of the AI model’s predictions regarding the likelihood of a bullish trend over the next 3 months. This percentage reflects how well the model forecasts future market movements.'
    },
    'fundamentalAnalysis': {
        'text': 'Accuracy of the AI model’s predictions regarding the likelihood of a bullish trend over the next 3 months. This percentage reflects how well the model forecasts future market movements.'
    },
    'score': {
        'text': 'Our AI model analyzes fundamental, technical, and statistical indicators to predict the probability of a bullish trend over the next 3 months. '
    },
    'currentRatio': {
        'text': 'The current ratio is used to measure a company`s short-term liquidity. A low number can indicate that a company will have trouble paying its upcoming liabilities.',
        'equation': 'Current Ratio = Current Assets / Current Liabilities'

    },
    'quickRatio': {
        'text': 'The quick ratio measure a company`s short-term liquidity. A low number indicates that the company may have trouble paying its upcoming financial obligations.',
        'equation': 'Quick Ratio = (Cash + Short-Term Investments + Accounts Receivable) / Current Liabilities'

    },
    'debtRatio': {
        'text': 'The Debt Ratio measures the proportion of a company’s total debt to its total assets. It provides insight into the company’s financial leverage and risk by showing how much of the company’s assets are financed through debt.',
        'equation': 'Debt Ratio = Total Debt / Total Assets'
    },
    'returnOnAssets': {
        'text': 'TReturn on assets (ROA) is a metric that measures how much profit a company is able to generate using its assets. It is calculated by dividing net income by the average total assets for the past 12 months.',
        'equation': 'ROA = (Net Income / Total Assets)'
    },
    'returnOnEquity': {
        'text': 'Return on Equity (ROE) measures a company’s profitability by comparing net income to shareholders’ equity. It reflects how effectively the company is using its equity base to generate profits for shareholders.',
        'equation': 'ROE = Net Income / Shareholders Equity'
    },
    'enterpriseValue': {
        'text': 'Enterprise value measures the total value of a company`s outstanding shares, adjusted for debt and levels of cash and short-term investments.',
        'equation': 'Enterprise Value = Market Cap + Total Debt - Cash & Equivalents - Short-Term Investments'
    },
    'freeCashFlowPerShare': {
        'text': 'Free cash flow per share is the amount of free cash flow attributed to each outstanding stock.',
    },
    'cashPerShare': {
        'text': 'Cash Per Share represents the amount of cash a company holds per share of its stock. It provides insight into the liquidity available to shareholders and the company’s ability to cover short-term obligations.',
    },
    'priceToFreeCashFlowsRatio': {
        'text': 'The price to free cash flow (P/FCF) ratio is similar to the P/E ratio, except it uses free cash flow instead of accounting earnings.',
        'equation': 'P/FCF Ratio = Market Capitalization / Free Cash Flow'
    },
    'sharesShort': {
        'text': 'Short Interest represents the total number of shares that have been sold short but not yet covered or closed out. It indicates the level of bearish sentiment among investors and the potential for a short squeeze if the stock price rises.',
    },
    'shortRatio': {
        'text': 'Short ratio is the ratio of shorted shares relative to the stock`s average daily trading volume. It estimates how many trading days it would take for all short sellers to cover their position.',
    },
    'shortFloatPercent': {
        'text': 'The percentage of a stock’s public float that has been sold short. This metric indicates the proportion of shares available for trading that have been sold short, reflecting investor sentiment and potential market pressure.'
    },
    'shortOutstandingPercent': {
        'text': 'The percentage of a stock’s publicly outstanding shares that have been sold short. It reflects the level of short-selling activity relative to the total number of shares available for trading.'
    },
    'failToDeliver': {
        'text': 'Fail to Deliver (FTD) represents the number of shares that were sold but not delivered to the buyer within the standard settlement period. This metric indicates issues in the settlement process and can signal potential liquidity or operational problems in the market.'
    },
    'relativeFTD': {
        'text': 'Relative FTD represents the proportion of Fail to Deliver (FTD) shares compared to the average trading volume. It indicates the percentage of undelivered shares in relation to typical volume.',
        'equation': 'Relative FTD = (Fail to Deliver Shares / Avg. Volume) * 100%'
    },
    'operatingCashFlowPerShare': {
        'text': 'Operating Cash Flow Per Share measures the amount of cash generated from operations per share of stock. It reflects the company’s ability to generate cash from its core business activities relative to the number of shares outstanding.',
    },
    'revenuePerShare': {
        'text': 'Revenue Per Share measures the amount of revenue generated by a company per share of its stock.',
    },
    'netIncomePerShare': {
        'text': 'Net Income Per Share shows the profit available to each outstanding share of stock, reflecting a company’s profitability and earnings potential.',
    },
    'shareholdersEquityPerShare': {
        'text': 'Shareholders Equity Per Share measures the equity available to shareholders per share, indicating the company net asset value and investment potential.',
    },
    'interestDebtPerShare': {
        'text': 'Interest Debt Per Share measures the interest expense allocated to each share of stock, indicating the company debt burden relative to its share count.',
    },
    'capexPerShare': {
        'text': 'CapEx Per Share measures capital expenditures allocated to each share, reflecting a company investment in growth and long-term assets relative to its share count.',
    },
    'grahamNumber': {
        'text': 'The Graham Number is a measure of a stock maximum fair value based on its earnings and book value. An ideal Graham Number indicates the stock maximum fair value; if the current share price is below this number, it may be a good buying opportunity, while a price above suggests it might be overvalued and a candidate for selling.',
        'equation': 'Graham Number = √(22.5 × Earnings Per Share × Book Value Per Share)'
    },
    'cashFlowToDebtRatio': {
        'text': 'The Cash Flow / Debt Ratio measures a company’s ability to cover its total debt with its operating cash flow. It provides insight into the company’s financial health and its capacity to manage debt obligations using cash generated from operations.',
    },
    'operatingCashFlowSalesRatio': {
        'text': 'The Operating Cash Flow / Sales Ratio measures the proportion of operating cash flow relative to total sales. It indicates how efficiently a company is converting its sales into cash flow from operations, reflecting its operational efficiency.',
    },
    'priceToOperatingCashFlowRatio': {
        'text': 'The Price to Cash Flow Ratio measures the price of a company’s stock relative to its operating cash flow per share. It helps assess whether the stock is overvalued or undervalued based on the cash flow it generates.',
    },
    'priceEarningsRatio': {
        'text': 'The Price to Earnings (PE) Ratio measures a company’s current share price relative to its earnings per share (EPS). It provides insight into how much investors are willing to pay for each dollar of earnings, helping to assess the stock’s valuation.',
    },
    'grossProfitMargin': {
        'text': 'Gross margin is the percentage of revenue left as gross profits, after subtracting cost of goods sold from the revenue.',
        'equation': 'Gross Margin = (Gross Profit / Revenue) * 100%'
    },
    'netProfitMargin': {
        'text': 'Profit margin is the percentage of revenue left as net income, or profits, after subtracting all costs and expenses from the revenue.',
        'equation': 'Profit Margin = (Net Income / Revenue) * 100%'
    },
    'pretaxProfitMargin': {
        'text': 'Pretax margin is the percentage of revenue left as profits before subtracting taxes.',
        'equation': 'Pretax Margin = (Pretax Income / Revenue) * 100%'
    },
    'ebitdaMargin': {
        'text': 'EBITDA margin is the percentage of revenue left as EBITDA, after subtracting all expenses except interest, taxes, depreciation and amortization from revenue.',
        'equation': 'EBITDA Margin = (EBITDA / Revenue) * 100%'
    },
    'assetTurnover': {
        'text': 'The asset turnover ratio measures the amount of sales relative to a company`s assets. It indicates how efficiently the company uses its assets to generate revenue.',
        'equation': 'Asset Turnover = Revenue / Average Assets'
    },
    'earningsYield': {
        'text': 'The earnings yield is a valuation metric that measures a company`s profits relative to stock price, expressed as a percentage yield. It is the inverse of the PE ratio.',
        'equation': 'Earnings Yield = (Earnings Per Share / Stock Price) * 100%'
    },
    'freeCashFlowYield': {
        'text': 'The free cash flow (FCF) yield measures a company`s free cash flow relative to its price, shown as a percentage. It is the inverse of the P/FCF ratio.',
        'equation': 'FCF Yield = (Free Cash Flow / Market Cap) * 100%'
    },
    'effectiveTaxRate': {
        'text': 'The effective tax rate is the percentage of taxable income paid in corporate income tax.',
        'equation': 'Effective Tax Rate = (Income Tax / Pretax Income) * 100%'
    },
    'fixedAssetTurnover': {
        'text': 'The Fixed Asset Turnover Ratio measures how efficiently a company utilizes its fixed assets to generate sales. It indicates the amount of revenue produced per dollar of fixed assets, reflecting the effectiveness of asset use in generating revenue.',
    },
    'sharesOutStanding': {
        'text': 'The total number of outstanding shares. If the company has many different types of shares, then this number assumes that all of the company`s stock is converted into the current share class. This number is used for calculating the company`s market cap.',
    },
    'employees': {
        'text': 'The company`s last reported total number of employees.',
    },
    'revenuePerEmployee': {
        'text': 'The amount of revenue that the company generates per each employee.',
        'equation': 'Revenue Per Employee = Total Revenue / Employee Count'
    },
    'profitPerEmployee': {
        'text': 'The amount of net income generated per each employee.',
        'equation': 'Profits Per Employee = Total Net Income / Employee Count'
    },
    'analystRating': {
        'text': 'The average rating of analysts for the stock.',
    },
    'topAnalystRating': {
        'text': 'The average rating of top analysts for the stock.',
    },
    'sector': {
        'text': 'The primary sector that the company operates in.',
    },
    'industry': {
        'text': 'The primary industry that the company operates in.',
    },
    'freeCashFlowMargin': {
        'text': 'FCF margin is the percentage of revenue left as free cash flow. FCF is calculated by subtracting capital expenditures (CapEx) from the operating cash flow (OCF). Both CapEx and OCF are shown on the cash flow statement.',
        'equation': 'FCF Margin = (Free Cash Flow / Revenue) * 100%'
    },
    'altmanZScore': {
        'text': 'The Altman Z-Score is a financial metric derived from a formula designed to predict the likelihood of a company facing bankruptcy within two years. A higher Z-Score indicates a lower probability of insolvency, while a lower Z-Score signals a higher risk of financial distress and potential bankruptcy.',
    },
    'piotroskiScore': {
        'text': 'The Piotroski F-Score is a score between 0 and 9 that determine the strength of a company`s financial position. The higher, the better.',
    },
    'totalAssets': {
        'text': 'Total assets is the sum of all current and non-current assets on the balance sheet. Assets are everything that the company owns.',
    },
    'workingCapital': {
        'text': 'Working capital is the amount of money available to a business to conduct its day-to-day operations. It is calculated by subtracting total current liabilities from total current assets.',
        'equation': 'Working Capital = Current Assets - Current Liabilities'
    },
    'shortTermDebtToCapitalization': {
        'text': 'Short-term debt to capitalization is a financial ratio that measures the proportion of a company’s short-term debt relative to its total capitalization. It indicates how much of the company’s capital structure is made up of short-term debt, which needs to be repaid within a year.',
    },
    'longTermDebtToCapitalization': {
        'text': 'Long-term debt to capitalization is a financial ratio that measures the proportion of a company’s long-term debt relative to its total capitalization. This ratio reflects the extent to which a company’s capital structure is funded by long-term debt, typically due after one year.',
    },
    'interestIncomeToCapitalization': {
        'text': 'Interest income to capitalization is a financial ratio that measures the proportion of a company’s interest income relative to its total capitalization. It helps assess how much of a company’s capital is being generated through interest-bearing assets, indicating reliance on interest income for funding its capital structure.',
    },
    'interestDebtPerShare': {
        'text': 'Interest debt per share is a financial metric that calculates the amount of interest-bearing debt attributed to each outstanding share of a company’s stock. It provides insight into the level of debt burden carried by each share, indicating potential financial risk to shareholders.',
    },
    "analystCounter": { 
        "text": "The number of analysts that have provided price targets and ratings for this stock."
    },
    "topAnalystCounter": { 
        "text": "The number of top analysts that have provided price targets and ratings for this stock."
    },
    "priceTarget": { 
        "text": "The average 12-month price target forecast predicted by wallstreet analysts."
    },
    "topAnalystPriceTarget": { 
        "text": "The average 12-month price target forecast predicted by top wallstreet analysts."
    },
    "upside": { 
        "text": "The difference between the price target forecast and the current price, expressed as a percentage."
    },
    "topAnalystUpside": { 
        "text": "The difference between the price target forecast by top analyst and the current price, expressed as a percentage."
    },
    "halalStocks": { 
        "text": "Halal-compliant stocks are identified by ensuring that a company's debt, interest income, and liquidity each remain below 30%. Additionally, companies involved in industries like alcohol, tobacco, gambling, and weapons are excluded to ensure adherence to Islamic principles."
    },
    "revenueGrowthYears": {
        "text": "For how many consecutive fiscal years the company's revenue has been growing.",
    },
    "epsGrowthYears": {
        "text": "For how many consecutive fiscal years the company's EPS has been growing.",
    },
    "netIncomeGrowthYears": {
        "text": "For how many consecutive fiscal years the company's net income has been growing.",
    },
    "grossProfitGrowthYears": {
        "text": "For how many consecutive fiscal years the company's gross profit has been growing.",
    },
    "ebit": {
        "text": "EBIT stands for Earnings Before Interest and Taxes and is a commonly used measure of earnings or profits. It is similar to operating income.",
        'equation': 'EBIT = Net Income + Interest + Taxes',
    },
    "priceToEarningsGrowthRatio": {
        "text": "The price/earnings to growth (PEG) ratio is calculated by dividing a company's PE ratio by its expected earnings growth next year.",
        "equation": "PEG Ratio = PE Ratio / Expected Earnings Growth"
    },
    "peg": {
        "text": "The price/earnings to growth (PEG) ratio is calculated by dividing a company's PE ratio by its expected earnings growth next year.",
        "equation": "PEG Ratio = PE Ratio / Expected Earnings Growth"
    },
    "evToSales": {
        "text": "The enterprise value to sales (EV/Sales) ratio is similar to the price-to-sales ratio, but the price is adjusted for the company's debt and cash levels.",
        "equation": "EV/Sales Ratio = Enterprise Value / Revenue"
    },
    "evToEarnings": {
        "text": "The enterprise value to earnings (EV/Earnings) ratio measures valuation, but the price is adjusted for the company's levels of cash and debt.",
        "equation": "EV/Earnings Ratio = Enterprise Value / Net Income"
    },
    "evToEBITDA": {
        "text": "The EV/EBITDA ratio measures a company's valuation relative to its EBITDA, or Earnings Before Interest, Taxes, Depreciation, and Amortization.",
        "equation": "EV/EBITDA Ratio = Enterprise Value / EBITDA"
    },
    "evToEBIT": {
        "text": "The EV/EBIT is a valuation metric that measures a company's price relative to EBIT, or Earnings Before Interest and Taxes.",
        "equation": "EV/EBIT Ratio = Enterprise Value / EBIT"
    },
    "evToFCF": {
        "text": "The enterprise value to free cash flow (EV/FCF) ratio is similar to the price to free cash flow ratio, except the price is adjusted for the company's cash and debt.",
        "equation": "EV/FCF Ratio = Enterprise Value / Free Cash Flow", 
    },
    "inventoryTurnover": {
        "text": "The inventory turnover ratio measures how many times inventory has been sold and replaced during a time period.",
        "equation": "Inventory Turnover Ratio = Cost of Revenue / Average Inventory", 
    },
    "ebitMargin": {
        "text": "EBIT Margin is a profitability ratio that measures the percentage of revenue left as EBIT (Earnings Before Interest and Taxes).",
        "equation": "EBIT Margin = (EBIT / Revenue) * 100%", 
    },
    "operatingMargin": {
        "text": "Operating margin is the percentage of revenue left as operating income, after subtracting cost of revenue and all operating expenses from the revenue.",
        "equation": "Operating Margin = (Operating Income / Revenue) * 100%", 
    },
    "sharesQoQ": {
        "text": "The change in the number of shares outstanding, comparing the most recent quarter to the previous quarter.",
    },
    "sharesYoY": {
        "text": "The change in the number of shares outstanding, comparing the most recent quarter to the same quarter a year ago.",
    },
    "floatShares": {
        "text": "Float is the amount of shares that are considered available for trading. It subtracts closely held shares by insiders and restricted stock from the total number of shares outstanding."
    },
    "interestCoverage": {
        "text": "The interest coverage ratio is a measure of the ability of a company to pay its interest expenses. It is calculated by dividing the company's Earnings Before Interest and Taxes (EBIT) by its interest expenses.",
        "equation": "Interest Coverage Ratio = EBIT / Interest Expense"
    },
    "date_expiration": {
        "text": "The expiration date indicates how long the option remains valid for exercise. This is particularly important when large investors (whales) are involved in short-term or long-term trades."
    },
    "execution_estimate": {
        "text": "Select the execution price relative to the bid and ask. The options are: 'Above Ask' (bullish), 'Below Bid' (bearish), 'At Ask' (neutral), 'At Bid' (neutral), and 'At Midpoint' (neutral)."
    },
    "moneyness": {
        "text": "Select the moneyness of the option. The options are: 'In the Money' (ITM) and 'Out of the Money' (OTM). ITM options have intrinsic value and OTM options are not yet profitable."
    },
    "moneynessPercentage": {
        "text": "Moneyness indicates how far an option's strike price is from the current stock price. It reflects whether an option is in the money (profitable if exercised now), at the money (strike equals stock price), or out of the money (not profitable). Moneyness is expressed as a percentage and calculated differently for calls and puts.",
        "equation": "Moneyness (Call) = ((Stock Price / Strike Price) - 1) × 100\nMoneyness (Put) = ((Strike Price / Stock Price) - 1) × 100"
    },
    "cost_basis": {
        "text": "It is the price at which the option was purchased and represents the total investment required to acquire the option. Large premiums (whale activity) can signal significant trading strategies, such as large bets on price movement or hedging strategies."
    },
    "put_call": {
        "text": "The contract type refers to whether the option is a 'Put' (betting on price decline) or a 'Call' (betting on price rise). This determines the direction of the trade."
    },
    "sentiment": {
        "text": "Sentiment is determined by the aggressor index. A value of 0.6 or higher indicates a bullish sentiment, below 0.5 indicates a bearish sentiment and 0.5 is considered neutral."
    },
    "volume": {
        "text": "The number of shares traded during the current or latest trading day."
    },
    "open_interest": {
        "text": "Open interest refers to the total number of outstanding option contracts that have not been settled. A rising open interest suggests growing market interest, while a decline may indicate reduced interest or closing positions."
    },
    "size": {
        "text": "Size refers to the number of contracts in a single trade or order. Larger sizes often indicate significant market activity and can be linked to institutional or whale trading."
    },
    "volumeOIRatio": {
        "text": "The Volume / Open Interest ratio compares the current trading volume to the total open interest. A higher ratio suggests increased market activity and potential short-term price movement, while a lower ratio indicates less activity relative to outstanding contracts."
    },
    "sizeOIRatio": {
        "text": "The Size / Open Interest Ratio compares the size of a single trade to the total open interest. A higher ratio may indicate that a large position is being taken relative to the existing contracts, suggesting significant market interest or potential for price movement."
    },
    "flowType": {
        "text": "Different Flow types such as Repeated flow, which identifies option trades that occur multiple times with the same characteristics (ticker, put/call, strike price, expiration). If the same trade appears more than three times, it's flagged as a repeated flow, indicating significant recurring interest in that specific option."
    },
    "option_activity_type": {
        "text": "The option activity type indicates the nature of the trade. A 'Sweep' occurs when an order is split across multiple exchanges to quickly execute, often signaling urgency. A 'Trade' refers to a standard option transaction executed on a single exchange."
    },
    "underlying_type": {
        "text": "The underlying type refers to the asset upon which the option is based. It can be an 'ETF' (Exchange-Traded Fund), which tracks a basket of assets, or a 'Stock,' which represents shares of a single company."
    },
    "callVolume": {
        "text": "Call volume refers to the total number of call option contracts traded during a given period. It indicates the level of interest or activity in call options, which grant the holder the right to buy the underlying asset at a specified price before expiration."
    },
    "putVolume": {
        "text": "Put volume refers to the total number of put option contracts traded during a given period. It indicates the level of interest or activity in put options, which grant the holder the right to sell the underlying asset at a specified price before expiration."
    },
    "gexRatio": {
        "text": "The GEX ratio or Gamma Exposure ratio, measures the sensitivity of the options market to changes in the price of the underlying asset. It is calculated by comparing the net gamma exposure of call and put options, providing insight into potential price stability or volatility."
    },
    "ivRank": {
        "text": "Implied Volatility (IV) Rank indicates how the current implied volatility compares to its historical range over a specific period. Expressed as a percentage, it helps traders determine whether options are relatively expensive (high IV Rank) or cheap (low IV Rank) compared to past volatility levels.",
        "equation": "IV Rank = ((Current IV - IV_min) / (IV_max - IV_min)) × 100"
    },
    "iv30d": {
        "text": "IV30d refers to the Implied Volatility over the past 30 days. It represents the market's expectations of the underlying asset's volatility over the next 30 days, as implied by the pricing of options, and is often used to gauge short-term market sentiment."
    },
    "totalOI": {
        "text": "Total Open Interest (OI), represents the total number of outstanding options contracts (both calls and puts) that have not been settled or exercised. It provides insight into the overall activity and liquidity in the options market for a particular asset."
    },
    "changeOI": {
        "text": "Change in Open Interest (Change OI) refers to the difference in the number of outstanding options contracts from one trading session to the next. A positive change indicates new positions are being opened, while a negative change suggests positions are being closed."
    },
    "changesPercentageOI": {
        "text": "Change in Open Interest (Change OI) refers to the difference in the number of outstanding options contracts from one trading session to the next. A positive change indicates new positions are being opened, while a negative change suggests positions are being closed."
    },
    "netCallPrem": {
        "text": "Net Call Premium (Net Call Prem) represents the net amount of premium paid for call options, calculated by subtracting the premium received from the premium paid. It provides insight into market sentiment and the demand for call options on the underlying asset."
    },
    "netPutPrem": {
        "text": "Net Put Premium (Net Put Prem) represents the net amount of premium paid for put options, calculated by subtracting the premium received from the premium paid. It indicates the demand for put options and can signal bearish market sentiment for the underlying asset."
    },
    "totalPrem": {
        "text": "Total Premium represents the total dollar value traded in option premiums, calculated by multiplying the option price by the volume for both calls and puts. It provides insight into the overall activity and sentiment in the options market for the underlying asset."
    },
    "pcRatio": {
        "text": "The Put/Call Ratio (P/C Ratio) measures the volume of put options traded relative to call options. A higher ratio suggests more bearish sentiment, while a lower ratio indicates more bullish sentiment, helping traders gauge market outlook and investor sentiment."
    },
    "dateExpiration": {  
        "text": "The expiration represents the expiration date of an options contract, indicating the last day the option can be exercised. Options lose value as they approach expiration, making this a crucial factor for traders assessing time decay and contract viability."
    },
    "optionType": {  
        "text": "The option type specifies whether an options contract is a Call or a Put. A Call option gives the holder the right to buy the underlying asset at a set price before expiration, while a Put option grants the right to sell, helping traders hedge risk or speculate on price movements."  
    },
    "strikePrice": {  
        "text": "The strike price is the predetermined price at which the holder of an options contract can buy (Call) or sell (Put) the underlying asset. It plays a key role in determining an option's intrinsic value and profitability relative to the market price."  
    },
    "assetType": {
        "text": "The asset type indicates the classification of the financial instrument. It helps differentiate between stocks, which represent ownership in individual companies, and ETFs (Exchange-Traded Funds), which are investment funds that hold a diversified portfolio of assets and trade on exchanges like individual stocks.",
    },
    "interestExpense": {
        "text": "Interest expense is the amount that the company paid in interest.",
    },
    "incomeTaxExpense": {
        "text": "Income tax is the amount of corporate income tax that the company has incurred during the fiscal period.",
    },
    "weightedAverageShsOut": {
        "text": "Basic shares outstanding is the total amount of common stock held by all of a company's shareholders.",
    },
    "weightedAverageShsOutDil": {
        "text": "Diluted shares outstanding is the total amount of common stock that will be outstanding if all stock options, warrants and convertible securities are exercised.",
    },
    "epsDiluted": {
        "text": "Earnings per share is the portion of a company's profit that is allocated to each individual stock. Diluted EPS is calculated by dividing net income by `diluted` shares outstanding.",
        "equation": "Diluted EPS = Net Income / Shares Outstanding (Diluted)",
    },
    "sellingGeneralAndAdministrativeExpenses": {
        "text": "Selling, general and administrative (SG&A) is an operating expense. It involves various company expenses that are not related to production."
    },
    "incomeBeforeTax": {
        "text": "Pretax income is a company's profits before accounting for income taxes.",
        "equation": "Pretax Income = Net Income + Income Taxes",
    },
    "sellingAndMarketingExpenses": {
        "text": "Selling & Marketing Expenses encompass the costs related to promoting and selling a company’s products or services. This includes advertising, sales commissions, promotional campaigns, and market research. These expenses are essential for driving revenue and enhancing brand visibility, yet they are not directly linked to the production process."
    },
    "otherExpenses": {
        "text": "Other Expenses encompass costs that fall outside the primary operational activities of the company. This category includes non-recurring items such as restructuring charges, asset impairments, legal settlements, and losses from discontinued operations. These expenses are reported separately to provide a clearer picture of the company's core operational performance."
    },
    "netCashProvidedByOperatingActivities": {
        "text": "Operating cash flow, also called cash flow from operating activities, measures the amount of cash that a company generates from normal business activities. It is the amount of cash left after all cash income has been received, and all cash expenses have been paid."
    },
    "capitalExpenditure": {
        "text": "Capital expenditures are also called payments for property, plants and equipment. It measures cash spent on long-term assets that will be used to run the business, such as manufacturing equipment, real estate and others."
    },
    "acquisitionsNet": {
        "text": "Cash paid for acquisitions, such as the acquisition of a company, a stake in a subsidiary, or a new product line."
    },
    "purchasesOfInvestments": {
        "text": "Purchase of Investments refers to the acquisition of financial assets like stocks or bonds by a company, typically for investment purposes. It represents an outflow of cash used to buy these investments."
    },
    "salesMaturitiesOfInvestments": {
        "text": "Sales Maturities of Investments signifies the selling or maturity of financial assets like stocks or bonds by a company. It represents an inflow of cash resulting from these investment activities."
    },
    "netDebtIssuance": {
        "text": "Debt Repayment is the process of paying off loans or debt obligations. It represents an outflow of cash as the company reduces its outstanding debt."
    },
    "netDividendsPaid": {
        "text": "The total amount paid out as cash dividends to shareholders."
    },
    "otherFinancingActivities": {
        "text": "Other financial activities includes miscellaneous financial transactions beyond regular operations that impact a company's cash flow."
    },
    "commonStockRepurchased": {
        "text": "The cash gained from issuing shares, or cash spent on repurchasing shares via share buybacks. A positive number implies that the company issued more shares than it repurchased. A negative number implies that the company bought back shares."
    },
    "netCashProvidedByInvestingActivities": {
        "text": "Investing cash flow is the total change in cash from buying and selling investments and long-term assets."
    },
    "deferredIncomeTax": {
        "text": "Deferred income tax refers to future tax liabilities or assets resulting from differences in how a company reports income for tax purposes versus financial reporting. It represents the amount of income tax that will be paid or saved in the future due to these differences."
    },
    "otherWorkingCapital": {
        "text": "Other working capital represents miscellaneous changes in short-term assets and liabilities that don't have specific categories, affecting the company's available cash."
    },
    "otherNonCashItems": {
        "text": "Other Non-Cash Items refers to non-cash transactions or adjustments that impact the company's financial performance but don't involve actual cash flows. These can include items like depreciation, amortization, or changes in the fair value of investments."
    },
    "depreciationAndAmortization": {
        "text": "Depreciation and amortization are accounting methods for calculating how the value of a business's assets change over time. Depreciation refers to physical assets, while amortization refers to intangible assets."
    },
    "netChangeInCash": {
        "text": "Net cash flow is the sum of the operating, investing and financing cash flow numbers. It is the change in cash and equivalents on the company's balance sheet during the accounting period. It is often shown as increase/decrease in cash and equivalents on the cash flow statement."
    },
    "otherInvestingActivities": {
        "text": "Other investing activities are investing activities that do not belong to any of the categories above. "
    },
    "changeInWorkingCapital": {
        "text": "Change in working capital is the difference between a company's short-term assets and liabilities over a specific period, reflecting how much cash flow is impacted by these changes."
    },
    "payoutFrequency": {
        "text": "Payout frequency is the schedule at which the dividends are paid. For example, a 'Quarterly' payout ratio implies that dividends are paid every three months."
    },
    "returnOnCapitalEmployed": {
        "text": "Return on capital employed (ROCE) is a financial ratio that measures a company's profitability and efficiency at deploying capital. The higher the value, the more effective a company is at generating profits from the capital it invests.",
        "equation": "ROCE = EBIT / (Total Assets - Current Liabilities)",
    },
    "stockBasedCompensationToRevenue": {
        "text": "The company's stock-based compensation expenses as a percentage of total revenue in the last 12 months."
    },
    "earningsDate": {
        "text": "The earnings date is the day that the company releases its quarterly earnings. Check the company's investor relations page to confirm the date."
    },
    "earningsTime": {
        "text": "The time that the company's earnings are released, either before market open or after market close."
    },
    "earningsRevenueEst": {
        "text": "The estimated revenue for the fiscal quarter being reported, based on a consensus of analysts."
    },
    "earningsEPSEst": {
        "text": "Estimated earnings per share (EPS) for the fiscal quarter being reported, based on analyst consensus."
    },
    "earningsRevenueGrowthEst": {
        "text": "The estimated year-over-year revenue growth for the fiscal quarter being reported, based on a consensus of analysts."
    },
    "earningsEPSGrowthEst": {
        "text": "The estimated year-over-year EPS growth for the fiscal quarter being reported, based on analyst consensus."
    },
    "grahamUpside": {
        "text": "The upside/downside for the stock price according to the Graham Number formula, which can be used to estimate the intrinsic value of a stock according to classical value investing principles. If the number is positive, the stock may be undervalued. If the number is negative, the stock may be overvalued."
    },
    "lynchFairValue": {
        "text": "The fair value formula associated with legendary investor Peter Lynch is used to estimate a company's intrinsic value based on its earnings per share and earnings growth rate. Earnings growth in the formula is capped at 25%. If it's less than 5%, then a value of 5% is used.",
        "equation": "Lynch Fair Value = 5-Year Earnings Growth * EPS",
    },
    "lynchUpside": {
        "text": "The upside/downside for the stock price according to the Peter Lynch Fair Value formula, which can be used to estimate a company's intrinsic value. If the number is positive, the stock may be undervalued. If the number is negative, the stock may be overvalued.",
    },
    "cashAndCashEquivalents": {
        "text": "Cash and short-term investments is the sum of 'Cash & Equivalents' and 'Short-Term Investments.' This is the amount of money that a company has quick access to, assuming that the cash equivalents and short-term investments can be sold at a short notice.",
        "equation": "Cash & Cash Equivalents = Cash & Equivalents + Short-Term Investments",
    },
    "operatingProfitMargin": {
        "text": "Operating margin is the percentage of revenue left as operating income, after subtracting cost of revenue and all operating expenses from the revenue.",
        "equation": "Operating Margin = (Operating Income / Revenue) * 100%",
    },
    "evToOperatingCashFlow": {
        "text": "The EV/EBIT is a valuation metric that measures a company's price relative to EBIT, or Earnings Before Interest and Taxes.",
        "equation": "EV/EBIT Ratio = Enterprise Value / EBIT",
    },
    "evToFreeCashFlow": {
        "text": "The enterprise value to free cash flow (EV/FCF) ratio is similar to the price to free cash flow ratio, except the price is adjusted for the company's cash and debt.",
        "equation": "EV/FCF Ratio = Enterprise Value / Free Cash Flow",
    },
    "priceToFreeCashFlowRatio": {
        "text": "The price to free cash flow (P/FCF) ratio is similar to the P/E ratio, except it uses free cash flow instead of accounting earnings.",
        "equation": "P/FCF Ratio = Market Capitalization / Free Cash Flow",
    },
    "netCash": {
        "text": "Net Cash / Debt is an indicator of the financial position of a company. It is calculated by taking the total amount of cash and cash equivalents and subtracting the total debt.",
        "equation": "Net Cash / Debt = Total Cash - Total Debt",
    },
        "retainedEarnings": {
        "text": "The portion of a company's net income that is retained and reinvested in the business rather than distributed as dividends. Positive retained earnings indicate accumulated profits over time, while negative retained earnings may suggest past losses or high dividend payouts."
    },
    "floatShare": {
        "text": "The number of a company's shares that are available for trading by the public, excluding closely held shares by insiders or major shareholders. A lower float can lead to higher volatility, while a higher float typically indicates more liquidity."
    },
    "debtToEquityRatio": {
        "text": "The debt-to-equity ratio measures a company's debt levels relative to its shareholders' equity or book value. A high ratio implies that a company has a lot of debt.",
        "equation": "Debt / Equity Ratio = Total Debt / Shareholders' Equity",
    },
    "debtToEBITDARatio": {
        "text": "The debt-to-EBITDA ratio is a company's debt levels relative to its trailing twelve-month EBITDA. A high ratio implies that debt is high relative to the company's earnings.",
        "equation": "Debt / EBITDA Ratio = Total Debt / EBITDA (ttm)",
    },
    "debtToFreeCashFlowRatio": {
        "text": "The debt-to-FCF ratio measures the debt levels relative to a company's free cash flow over the previous twelve months. If the ratio is high, it means that the company will need to spend a lot of the cash it generates on paying back debt.",
        "equation": "Debt / FCF Ratio = Total Debt / Free Cash Flow (ttm)",
    },
    "interestCoverageRatio": {
        "text": "The interest coverage ratio is a measure of the ability of a company to pay its interest expenses. It is calculated by dividing the company's Earnings Before Interest and Taxes (EBIT) by its interest expenses.",
        "equation": "Interest Coverage Ratio = EBIT / Interest Expense",
    },
    "lastStockSplit": {
        "text": "The date when the company last performed a stock split.",
    },
    "splitType": {
        "text": "There are two types of stock splits: Forward and Reverse. Forward means that the share count increases and the stock price goes down. Reverse means that the stock count decreases and the stock price goes up.",
    },
    "splitRatio": {
        "text": "The split ratio, also called the split factor, is the ratio in which the amount of shares changes because of the stock split.",
    },
    "cagrNext3YearEPS": {
        "text": "The forecasted growth in annual earnings per share (EPS) over the next 3 years according to stock analysts, calculated as the compound annual growth rate (CAGR).",
    },
    "cagrNext5YearEPS": {
        "text": "The forecasted growth in annual earnings per share (EPS) over the next 5 years according to stock analysts, calculated as the compound annual growth rate (CAGR).",
    },
    "cagrNext3YearRevenue": {
        "text": "The forecasted growth in annual revenue over the next 3 years according to stock analysts, calculated as the compound annual growth rate (CAGR)."
    },
    "cagrNext5YearRevenue": {
        "text": "The forecasted growth in annual revenue over the next 5 years according to stock analysts, calculated as the compound annual growth rate (CAGR)."
    },
    "changesPercentage": {
        "text": "The percentage change in the stock price on the current or latest trading day."
    },

}


save_json(data)

