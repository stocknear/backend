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
    'debtEquityRatio': {
        'text': 'The debt-to-equity ratio measures a company`s debt levels relative to its shareholders equity or book value. A high ratio implies that a company has a lot of debt.',
        'equation': 'Debt / Equity Ratio = Total Debt / Shareholders Equity'

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
    'priceCashFlowRatio': {
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




}


save_json(data)

