import argparse
import sys
from datetime import datetime
import orjson
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TEST_MODE = False
SYMBOL = "CRH"

def save_json(data,  quarter, fiscal_year, version):
    dir_path = "json/blog"
    os.makedirs(dir_path, exist_ok=True)  # Create directory if it doesn't exist
    file_path = os.path.join(dir_path, f"{SYMBOL}-{quarter}-{fiscal_year}-v{version}.json")
    with open(file_path, 'wb') as file:
        file.write(orjson.dumps(data))


def get_text_instructions():
    return """
        Based on the provided financial metric data for a company—where each item includes a label, a key, a numeric value, and a sentiment (e.g., "Very Good", "Average")—write a professional but still relatable and insightful financial summary similar in tone and structure to an equity analyst’s report.

        The summary should:

        • Begin with a general overview of the company’s financial or operational profile.
        • Use clear, descriptive subheadings in the text for improved search visibility (e.g., "Key Strengths: High ROE", "Areas to Watch: SBC Ratio").
        • Naturally integrate relevant long-tail keywords like "pre-earnings review", "upcoming earnings analysis", and "financial performance snapshot".

        The summary must:

        1. Highlight key strengths (e.g., high return metrics) in plain text.
        2. Discuss average or weak areas, explaining implications, under subheadings like "Areas to Watch".
        3. Conclude with a forward-looking outlook statement, avoiding generic closing phrases.

        Formatting and Tone:
        • Wrap each paragraph in <p class="mb-4">…</p> for consistency.
        • Keep tone balanced, professional, and reader-friendly, aimed at investors and stakeholders.
        • Ensure SEO by using target keywords at least once without keyword stuffing.
        """

# Existing functions unchanged

def get_summary_instructions():
    return """
        You’re writing a concise, equity‑analyst‑style summary—akin to the company example—using a list of financial metrics. Each metric entry provides:
          • label (e.g. “Price to Earnings (P/E)”)  
          • key (e.g. “PE”)  
          • numeric value  
          • sentiment (“Very Good”, “Average”, etc.)

        Follow this structure:

        1. **Valuation Overview**  
           • Compare current multiples (P/E, P/FCF, P/S, PEG, etc.) to historical or peer averages.  
           • Interpret what elevated or depressed ratios imply about investor expectations or over/under‑valuation.

        2. **Growth & Profitability**  
           • Summarize recent trends in top‑line (revenue) and bottom‑line (gross profit, operating income, net income, FCF).  
           • Highlight whether improving margins or cash flows offset any revenue headwinds.

        3. **Financial Health & Capital Allocation**  
           • Note liquidity (current ratio), leverage (debt levels), and any share buybacks or dividends.  
           • Discuss the implications for balance‑sheet strength and management confidence.

        4. **Efficiency & Management Quality**  
           • Cite return metrics (ROE, ROA, ROIC) and SBC rates.  
           • Explain how these reflect cost control, pricing power, and potential dilution.

        5. **Forward‑Looking Takeaway**  
           • Weave together valuation, growth, health, and efficiency to sketch the company’s near‑term outlook.  
           • Avoid “In conclusion” or “In summary”; instead, craft a closing that nods to long‑term trajectory and trade‑offs.

        **Formatting & Tone**  
        • Write 3–5 paragraphs, each wrapped in HTML:  
          ```html
          <p class="mb-4">…</p>
          <h3 class="text-xl sm:text-2xl font-bold mb-4">…</h3>
          ```  
        • Maintain a balanced, professional yet approachable tone—insightful, not promotional or overly critical.
        ALWAYS USE THE CORRECT html style for each paragraph: <p class="mb-4">...closing paragraph...</p>
        """

def get_seo_title_desc_instructions():
    return """
        Given the summary text of a blog post, generate a highly optimized SEO title and meta description that drive click-through rates and target relevant keywords.

        The instructions should cover:

        1. **Keyword Extraction:**
           • Identify 2–3 primary and secondary keywords or phrases from the summary.
           • Use a mix of short-tail and long-tail keywords reflecting user intent.

        2. **Title Crafting Guidelines:**
           • Length: Aim for 50–60 characters (including spaces).
           • Keyword placement: Include the primary keyword near the beginning.
           • Emotional triggers: Use action verbs or power words to increase engagement.
           • Clarity & Promise: Clearly convey the main benefit or takeaway.
           • Uniqueness: Avoid clickbait; ensure the title accurately reflects the content.

        3. **Meta Description Guidelines:**
           • Length: Aim for 150–160 characters.
           • Keyword usage: Incorporate the primary keyword once and a secondary keyword if possible.
           • Call-to-action: Include a subtle CTA (e.g., "Learn how", "Discover why").
           • Benefit statement: Highlight the value or insight the reader will gain.

        4. **Optimization Checklist:**
           • Avoid keyword stuffing and punctuation overuse (e.g., excessive exclamation marks).
           • Ensure readability and flow for human users and search engines.
           • Match tone and style to the blog’s brand voice.

        5. **Output Format:**
           • Return a JSON or dictionary with two keys: `title` and `description`.

        Example:
        ```json
        {
          "title": "How to Boost Your Productivity: 5 Proven Time-Management Strategies",
          "description": "Discover 5 actionable time-management techniques to skyrocket your productivity and reclaim your workday. Learn the secrets top performers use."
        }
        ```

        Use these instructions to automatically generate SEO-optimized titles and descriptions that improve search visibility and click-through rates.  
        """

def get_agent_instructions():
    return """
        You are a senior financial analyst whose job is to explain company data in a way that’s simple, clear, and relatable — without oversimplifying or losing the important points. Your audience may include beginners, so speak plainly, avoid jargon where possible, and break things down step by step.
        You're output must not be detected as "AI generated text" otherwise you will die. This is a high stake situation where you have to perform the best otherwise you die.
        Your output must strictly follow this HTML format: wrap each paragraph in <p class="mb-4">...</p>. Do not use Markdown, bullet points, or plain text formatting.

        Your key goals are:

        1. Clarity first. Always explain what a number means in practical terms. Don’t just say, “Net income increased by 10%.” Instead, say something like, “The company made 10% more profit compared to last quarter, which suggests it’s managing costs better or growing sales.”

        2. Avoid overused phrases. Don’t say things like "Let's delve into..." or "In this deep dive..." Just say, "Here's what the data shows" or "This means..."

        3. Keep it human. Speak like a real person talking to someone across the table. Think of how you'd explain something to a friend who’s smart but not in finance. Use examples when helpful.

        4. Highlight what matters. Focus on what actually affects the business — revenue trends, margins, cash flow, debt levels, and anything unusual. If something is strong or weak, say so directly.

        5. No fluff. If a number or section doesn’t say much, don’t force an explanation. It’s fine to say, “This part hasn’t changed much and isn’t a big factor right now.”

        6. Stay professional. Be friendly and engaging, but avoid being too casual or using slang. You're still speaking as a trusted expert.

        Example output format:

        <p class="mb-4">Revenue grew by 12% this quarter, mainly due to strong sales in the North American market. This shows the company’s strategy in that region is working well.</p>

        <p class="mb-4">However, profit margins were slightly lower, likely because of rising supply chain costs. That’s something to keep an eye on next quarter.</p>

        Always aim to give people useful takeaways, not just numbers.
        ALWAYS USE THE CORRECT html style <p class="mb-4">...closing paragraph...</p> <h3 class="text-xl sm:text-2xl font-bold mb-4">…</h3>
        """


INSTRUCTIONS = get_agent_instructions()


def get_llm_output(user_query: str) -> list | None:
    try:
        response = client.chat.completions.create(
            model=os.getenv("REASON_CHAT_MODEL"),
            messages=[
                {"role": "system", "content": "INSTRUCTIONS"},
                {"role": "user", "content": user_query}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return None

def get_sentiment(value):
    if value is None:
        return "Very Bad"
    elif value < 0:
        return "Very Bad"
    elif value < 5:
        return "Bad"
    elif value < 15:
        return "Average"
    elif value < 25:
        return "Good"
    else:
        return "Very Good"

def get_sentiment_growth(value):
    if value is None:
        return "Very Bad"
    elif value < -10:
        return "Very Bad"
    elif value < 0:
        return "Bad"
    elif value < 10:
        return "Average"
    elif value < 30:
        return "Good"
    else:
        return "Very Good"

def get_avg_sentiment(value):
    if value is None:
        return "Very Bad"
    elif value < -30:
        return "Very Good"
    elif value < -15:
        return "Good"
    elif value < 0:
        return "Average"
    elif value < 20:
        return "Bad"
    else:
        return "Very Bad"

def calculate_growth(current, previous):
    if current is None or previous is None or previous == 0:
        return None
    return ((current - previous) / abs(previous)) * 100


def get_cumulative_returns(SYMBOL):

    with open(f"json/historical-price/max/{SYMBOL}.json", "rb") as file:
        SYMBOL_price = orjson.loads(file.read())[-252:]
    initial_close = SYMBOL_price[0]['close']

    # Load SPY benchmark data
    with open("json/historical-price/max/SPY.json", "rb") as file:
        spy_price = orjson.loads(file.read())[-252:]
    initial_spy_close = spy_price[0]['close']

    # Calculate cumulative returns for both the stock and SPY benchmark
    cumulative_returns = []
    for i in range(len(SYMBOL_price)):
        try:
            date = SYMBOL_price[i]['time']
            
            # Stock cumulative return
            close = SYMBOL_price[i]['close']
            cumulative_roi = round(((close / initial_close) - 1) * 100, 2)
            
            # SPY cumulative return
            spy_close = spy_price[i]['close']
            cumulative_spy = round(((spy_close / initial_spy_close) - 1) * 100, 2)
            
            # Append combined result
            cumulative_returns.append({
                "date": date,
                "cumulativeTicker": cumulative_roi,
                "cumulativeBenchmark": cumulative_spy
            })
        except Exception as e:
            # In case of any error, you could log the exception if needed
            pass

    # Example output
    return cumulative_returns

def get_overview( screener_data):
    res = {}
    with open(f"json/profile/{SYMBOL}.json","rb") as file:
        data = orjson.loads(file.read())
        
        if TEST_MODE:
            res['description'] = data['description']
        else:
            user_query = f"Summarize the text: {data['description']}"
            res['description'] = get_llm_output(user_query)
    
    with open(f"json/quote/{SYMBOL}.json", 'r') as file:
        data = orjson.loads(file.read())
        res['marketCap'] = data['marketCap']
        dt = datetime.strptime(data['earningsAnnouncement'], '%Y-%m-%dT%H:%M:%S.%f%z')
        res['nextEarning'] = dt.strftime("%B %d, %Y")
        res['epsTTM'] = data['eps']
        res['peTTM'] = data['pe']
    
    res['annualDividend'] = screener_data.get('annualDividend',None)
    res['dividendYield'] = screener_data.get('dividendYield',None)
    res['priceToSalesRatio'] = screener_data.get('priceToSalesRatio',None)
    res['priceToBookRatio'] = screener_data.get('priceToBookRatio',None)
    res['sharesOutstanding'] = screener_data.get('sharesOutStanding',None)
    res['shortFloatPercent'] = screener_data.get('shortFloatPercent',None)
    res['shortOutstandingPercent'] = screener_data.get('shortOutstandingPercent',None)
    res['forwardPE'] = screener_data.get('forwardPE',None)
    res['sector'] = screener_data.get('sector',None)

    res['cumulativeReturns'] = get_cumulative_returns(SYMBOL)
    return res




def get_financial_health( screener_data):
    fields = [
        ("Gross Profit Margin", "grossProfitMargin"),
        ("Operating Profit Margin", "operatingProfitMargin"),
        ("Net Margin", "netProfitMargin"),
        ("FCF Margin", "freeCashFlowMargin"),
        ("EBITDA Margin", "ebitdaMargin"),
    ]

    res = []
    for label, key in fields:
        value = screener_data.get(key)
        sentiment = get_sentiment(value)
        res.append({
            "label": label,
            "value": value,
            "sentiment": sentiment
        })

    if TEST_MODE:
        text = ""
    else:
        user_query = f"Follow the instruction for the company {SYMBOL} with the data {res}: {get_text_instructions()}"
        text = get_llm_output(user_query)
    return {'data': res, 'text': text}



def get_growth():
    # Define the metrics in a way that is easy to extend
    metrics = [
        ("Revenue Growth", "revenue", "income-statement"),
        ("Gross Profit Growth", "grossProfit", "income-statement"),
        ("Operating Income Growth", "operatingIncome", "income-statement"),
        ("Net Income Growth", "netIncome", "income-statement"),
        ("Free Cash Flow Growth", "freeCashFlow", "cash-flow-statement"),
        ("Operating Cash Flow Growth", "freeCashFlow", "cash-flow-statement"),
    ]

    # Cache loaded data by statement type
    data_cache = {}

    summary = []

    for label, key, statement_type in metrics:
        # Load and cache the data for each statement type only once
        if statement_type not in data_cache:
            with open(f"json/financial-statements/{statement_type}/annual/{SYMBOL}.json", "rb") as file:
                data_cache[statement_type] = orjson.loads(file.read())

        current = data_cache[statement_type][0]
        previous = data_cache[statement_type][1]

        growth = calculate_growth(current.get(key), previous.get(key))
        sentiment = get_sentiment_growth(growth)

        summary.append({
            "label": label,
            "value": round(growth, 2) if growth is not None else None,
            "sentiment": sentiment
        })

    if TEST_MODE:
        text = ""
    else:
        user_query = f"Follow the instruction for the company {SYMBOL} with the data {summary}: {get_text_instructions()}"
        text = get_llm_output(user_query)
    return {'data': summary, 'text': text}

    return summary


def get_valuation(screener_data):
    
    keys = ['priceToEarningsRatio', 'priceToFreeCashFlowRatio', 'priceToSalesRatio','priceToBookRatio','priceToEarningsGrowthRatio']

    # Load the ratio data
    with open(f"json/financial-statements/ratios/annual/{SYMBOL}.json", "rb") as file:
        data = orjson.loads(file.read())

    # Ensure we have at least 5 years of data (excluding the most recent one)
    last_5_years = data[1:6]

    result = {}

    for key in keys:
        # Extract last 5 years of the specified key
        historical_values = [entry.get(key) for entry in last_5_years if entry.get(key) is not None]

        if not historical_values:
            result[key] = {
                "error": f"No valid {key} values found"
            }
            continue

        # Calculate the average of the last 5 years
        avg_value = sum(historical_values) / len(historical_values)
        latest_value = screener_data.get(key)

        if latest_value is None:
            result[key] = {
                "error": f"No latest {key} value found"
            }
            continue

        # Calculate upside/downside
        if avg_value == 0:
            upside = None  # or float('inf') or custom handling for division by zero
        else:
            upside = round((latest_value - avg_value) / abs(avg_value) * 100, 2)

        sentiment = get_avg_sentiment(upside)

        result[key] = {
            "fiveYearAvg": round(avg_value,2),
            "value": round(latest_value,2),
            "upside": upside,
            "sentiment": sentiment
        }

    if TEST_MODE:
        text = ""
    else:
        user_query = f"Follow the instruction for the company {SYMBOL} with the data {result}: {get_text_instructions()}"
        text = get_llm_output(user_query)
    return {'data': result, 'text': text}
        
def get_industry( screener_data):
    result = []
    industry = screener_data.get('industry')

    keys = [
        'evToFreeCashFlow', 'evToEBIT', 'evToEBITDA', 'priceToFreeCashFlowRatio',
        'priceToSalesRatio', 'priceToEarningsRatio', 'cagr5YearRevenue',
        'cagr5YearEPS', 'revenuePerShare', 'grossProfitMargin', 'netProfitMargin',
        'operatingProfitMargin', 'revenuePerEmployee', 'altmanZScore',
        'returnOnInvestedCapital', 'returnOnEquity', 'returnOnInvestedCapital'
    ]

    labels = {
        'evToFreeCashFlow': 'EV/FCF',
        'evToEBIT': 'EV/EBIT',
        'evToEBITDA': 'EV/EBITDA',
        'priceToFreeCashFlowRatio': 'P/FCF',
        'priceToSalesRatio': 'P/S',
        'priceToEarningsRatio': 'P/E',
        'cagr5YearRevenue': '5Y Revenue CAGR',
        'cagr5YearEPS': '5Y EPS CAGR',
        'revenuePerShare': 'Revenue/Share',
        'grossProfitMargin': 'Gross Margin',
        'netProfitMargin': 'Net Margin',
        'operatingProfitMargin': 'Operating Margin',
        'revenuePerEmployee': 'Revenue/Employee',
        'altmanZScore': 'Altman Z-Score',
    }

    with open(f"json/average/industry/data.json") as file:
        industry_avg = orjson.loads(file.read())[industry]


    for key in keys:
        avg_value = industry_avg.get(key)
        latest_value = screener_data.get(key)

        if avg_value is None or latest_value is None:
            continue  # Skip if data is missing

        label = labels.get(key, key)  # Fallback to key if label is missing

        upside = round((latest_value - avg_value) / abs(avg_value) * 100, 2) if avg_value != 0 else None

        result.append({
            "label": label,
            "key": key,
            "industryAvg": round(avg_value, 2),
            "value": round(latest_value, 2),
            "upside": upside,
        })

    if TEST_MODE:
        text_1 = ""
        text_2 = ""
        text_3 = ""
    else:
        table_1 = []
        table_2 = []
        table_3 = []
        for item in result:
            if item['key'] in ['evToFreeCashFlow','evToEBIT','evToEBITDA','priceToEarningsRatio','priceToSalesRatio','priceToFreeCashFlowRatio']:
                table_1.append(item)
            elif item['key'] in ['cagr5YearRevenue','cagr5YearEPS','revenuePerShare','revenuePerEmployee']:
                table_2.append(item)
            elif item['key'] in ['altmanZScore','operatingProfitMargin','netProfitMargin','grossProfitMargin']:
                table_3.append(item)
        
        user_query = f"Follow the instruction for the company {SYMBOL} with the context of Industry Average. The data {table_1}: {get_text_instructions()}"
        text_1 = get_llm_output(user_query)

        user_query = f"Follow the instruction for the company {SYMBOL} with the context of Industry Average. The data {table_2}: {get_text_instructions()}"
        text_2 = get_llm_output(user_query)

        user_query = f"Follow the instruction for the company {SYMBOL} with the context of Industry Average. The data {table_3}: {get_text_instructions()}"
        text_3 = get_llm_output(user_query)

    return {'data': result, 'textOne': text_1, 'textTwo': text_2, 'textThree': text_3}


def get_price_reaction():
    with open(f"json/earnings/past/{SYMBOL}.json","rb") as file:
        data = orjson.loads(file.read())

    stats = data.get('stats')
    history = data.get('history')
    next_day_changes = [
        item.get("forward_2_days_change_percent")
        for item in history
        if item.get("forward_2_days_change_percent") is not None
    ]

    # Average price impact
    if next_day_changes:
        avg_price_impact = sum(next_day_changes) / len(next_day_changes)
    else:
        avg_price_impact = 0

    # Volatility impact (average absolute range)
    total_range = 0
    valid_items = 0

    for item in history:
        high = item.get("high")
        low = item.get("low")
        close = item.get("close")
        if high is not None and low is not None and close:
            range_percent = ((high - low) / close) * 100
            total_range += range_percent
            valid_items += 1

    volatility_impact = total_range / valid_items if valid_items else 0

    result = {**stats, "avgPriceImpact": round(avg_price_impact, 1),"volatilityImpact": round(volatility_impact, 1)}

    if TEST_MODE:
        text = ""
    else:
        user_query = f"Follow the instruction for the company {SYMBOL} with the context of Price Reaction for Earnings Day. The data {result}: {get_text_instructions()}"
        text = get_llm_output(user_query)

    return {'data': result, 'text': text}


def get_management( screener_data):
    result = []

    # Extract necessary values from screener_data
    industry = screener_data.get('industry')
    sbc = screener_data.get('stockBasedCompensation')
    revenue = screener_data.get('revenue')
    operating_cash_flow = screener_data.get('operatingCashFlow')
    free_cash_flow = screener_data.get('freeCashFlow')

    # Safely compute SBC ratios (avoiding division by zero)
    def safe_ratio(numerator, denominator):
        return (numerator/denominator)*100 if numerator is not None and denominator else None

    sbc_to_revenue = safe_ratio(sbc, revenue)
    sbc_to_ocf = safe_ratio(sbc, operating_cash_flow)
    sbc_to_fcf = safe_ratio(sbc, free_cash_flow)

    # Add the computed ratios into screener_data for unified processing
    screener_data = screener_data.copy()  # avoid mutating original
    screener_data.update({
        "sbcToRevenueRatio": sbc_to_revenue,
        "sbcToOperatingCashFlowRatio": sbc_to_ocf,
        "sbcToFreeCashFlowRatio": sbc_to_fcf
    })

    # Define the keys and corresponding labels
    keys = [
        'sbcToRevenueRatio',
        'sbcToOperatingCashFlowRatio',
        'sbcToFreeCashFlowRatio',
        "returnOnEquity",
        "returnOnAssets",
        "returnOnInvestedCapital",
        "returnOnCapitalEmployed",
    ]

    labels = {
        'sbcToRevenueRatio': "SBC as % of Revenue",
        'sbcToOperatingCashFlowRatio': "SBC as % of Operating Cash Flow",
        'sbcToFreeCashFlowRatio': 'SBC as % of Free Cash Flow',
        "returnOnEquity": "Return on Equity",
        "returnOnAssets": "Return on Assets",
        "returnOnInvestedCapital": "Return on Invested Capital",
        "returnOnCapitalEmployed": "Return on Capital Employed",
    }

    for key in keys:
        latest_value = screener_data.get(key)
        label = labels.get(key, key)

        # You may define `previous` elsewhere, otherwise set growth to 0
        sentiment = get_sentiment_growth(latest_value)
        result.append({
            "label": label,
            "key": key,
            "value": round(latest_value, 2),
            "sentiment": sentiment,
        })

    if TEST_MODE:
        text=""
    else:
        user_query = f"Follow the instruction for the company {SYMBOL} with the context of how good the Management is. The data {result}: {get_text_instructions()}"
        text = get_llm_output(user_query)

    return {'data': result, 'text': text}


def get_summary( data):
    if TEST_MODE:
        text =""
    else:
        user_query = f"Write a good summary for my blog post for the company {SYMBOL}. The context data is data {data}: {get_summary_instructions()}"
        text = get_llm_output(user_query)

    return text

def get_seo_title_description( data, earnings_date, quarter, fiscal_year):
    if TEST_MODE:
        text =""
    else:
        user_query = (
        f"Generate an SEO‑optimized title and meta description for my blog post about {SYMBOL}'s upcoming earnings preview "
        f"({quarter} {fiscal_year}) on {earnings_date}, using this financial summary: {data}. "
        f"{get_seo_title_desc_instructions()}")

        text = get_llm_output(user_query)
    return text

def main():
    version = "1.0"
    res = {}
    
    
    try:
        with open(f"json/earnings/next/{SYMBOL}.json","rb") as file:
            earnings_data = orjson.loads(file.read())
        with open(f"json/earnings/raw/{SYMBOL}.json","rb") as file:
            data = orjson.loads(file.read())[0] #next quarter and fiscal year
            quarter = data.get('period')
            fiscal_year = data.get('period_year')
            earnings_date = data.get('date')
            print(earnings_date)
    except:
        print("No earnings data found.")
        return

    if os.path.exists(f"json/blog/{SYMBOL}-{quarter}-{fiscal_year}-v{version}.json"):
        print(f'Blog post for {SYMBOL} {quarter}-{fiscal_year}-v{version} already exist!')
        return

    if earnings_data and quarter and fiscal_year:
        res['nextEarningsData'] = earnings_data

        step = 1
        total_steps = 8

        with open("json/stock-screener/data.json", "rb") as file:
            stock_screener_data = orjson.loads(file.read())

        stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}
        screener_data = stock_screener_data_dict.get(SYMBOL)


        res['overview'] = get_overview( screener_data)
        res['name'] = screener_data.get('name', None)
        res['symbol'] = screener_data.get('symbol', None)
        print(f"Overview Done! {step}/{total_steps}")
        step += 1

        res['financialHealth'] = get_financial_health( screener_data)
        print(f"Financial Health Done! {step}/{total_steps}")
        step += 1

        res['growth'] = get_growth()
        print(f"Growth Done! {step}/{total_steps}")
        step += 1

        res['valuation'] = get_valuation(screener_data)
        print(f"Valuation Done! {step}/{total_steps}")
        step += 1

        res['industry'] = get_industry(screener_data)
        print(f"Industry Done! {step}/{total_steps}")
        step += 1

        res['priceReaction'] = get_price_reaction()
        print(f"Price Reaction Done! {step}/{total_steps}")
        step += 1

        res['management'] = get_management( screener_data)
        print(f"Management Done! {step}/{total_steps}")
        step += 1

        res['summary'] = get_summary( screener_data)
        print(f"Summary Done! {step}/{total_steps}")

        print("#=========SEO Meta Data============#")
        print(get_seo_title_description( res['summary'], earnings_date, quarter, fiscal_year))

        save_json(res,  quarter, fiscal_year, version)


    else:
        print("No earnings data found.")


  

if __name__ == "__main__":
    main()
