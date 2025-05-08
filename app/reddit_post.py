import praw
import orjson
from datetime import datetime
import os
from dotenv import load_dotenv
import time
import ujson
import argparse


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Post Type')
    parser.add_argument('--post_type', choices=['earnings', 'stock-list', 'premarket', 'aftermarket',"option"], 
                        type=str, default='earnings',
                        help='Post type: "earnings" (default), "stock-list", "premarket", or "aftermarket"')
    return parser.parse_args()


def get_current_weekday():
    """Return the current weekday name."""
    return datetime.now().strftime("%A")

def format_time(time_str):
    """Format time string to AM/PM format"""
    if not time_str:
        return ""
    
    try:
        time_parts = time_str.split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        
        period = "AM" if hours < 12 else "PM"
        if hours > 12:
            hours -= 12
        elif hours == 0:
            hours = 12
            
        return f"{hours:02d}:{minutes:02d} {period}"
    except:
        return ""


def format_number(num, decimal=False):
    """Abbreviate large numbers with B/M suffix"""
    if decimal:
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        return f"{num:,.0f}"
    else:
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:,.2f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:,.2f}M"
        return f"{num:,.0f}"  # Format smaller numbers with commas


def calculate_yoy_change(current, prior):
    """Calculate year-over-year percentage change"""
    if prior and prior != 0:
        return ((current / prior - 1) * 100)
    return 0

def get_market_timing(time_str):
    """Determine if earnings are before, after, or during market hours"""
    if not time_str:
        return ""
    
    try:
        time_parts = time_str.split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        
        if hours < 9 or (hours == 9 and minutes <= 30):
            return "before market opens."
        elif hours >= 16:
            return "after market closes."
        else:
            return "during market."
    except:
        return ""

def format_upcoming_earnings_data(earnings_data):
    """Format earnings data into Reddit-friendly markdown with hyperlinks."""
    formatted_items = []
    
    for item in earnings_data:
        symbol = item.get('symbol', None)
        if symbol is not None:
            name = item.get('name', 'Unknown')
            market_timing = get_market_timing(item.get('time'))
            revenue_formatted = format_number(item.get('revenueEst', 0))
            revenue_yoy = calculate_yoy_change(item.get('revenueEst', 0), item.get('revenuePrior', 1))  # Avoid division by zero
            eps_yoy = calculate_yoy_change(item.get('epsEst', 0), item.get('epsPrior', 1))  # Avoid division by zero

            # Determine reporting time text
            if item.get('isToday'):
                report_timing = "will report today"
            else:
                current_day = datetime.now().strftime('%A')
                report_timing = "will report tomorrow" if current_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday'] else "will report Monday"
            
            # Create hyperlink for symbol
            symbol_link = f"[{symbol}](https://stocknear.com/stocks/{symbol})"

            # Format the entry text
            entry = (
                f"* **{name}** ({symbol_link}) {report_timing} {market_timing} "
                f"Analysts estimate {revenue_formatted} in revenue ({revenue_yoy:.2f}% YoY) and "
                f"${item.get('epsEst', 0):.2f} in earnings per share ({eps_yoy:.2f}% YoY).\n\n"
            )
            formatted_items.append(entry)
    
    support_message = "\nInvest in yourself and embrace data-driven decisions to minimize losses, identify opportunities and achieve consistent growth with [Stocknear](https://stocknear.com/pricing) 🚀"
    
    return "".join(formatted_items) + support_message


def format_recent_earnings_data(earnings_data):
    """Format earnings data into Reddit-friendly markdown with bullet points."""
    formatted_items = []
    
    for item in earnings_data:
        symbol = item.get('symbol', None)
        if symbol is not None:
            name = item.get('name', 'Unknown')
            time = format_time(item.get('time', ''))
            
            # Financial calculations
            revenue = item.get('revenue', 0)  # Changed from revenueEst to revenue for actual results
            revenue_prior = item.get('revenuePrior', 1)
            revenue_surprise = item.get('revenueSurprise', 0)
            eps = item.get('eps', 0)  # Changed from epsEst to eps for actual results
            eps_prior = item.get('epsPrior', 1)
            eps_surprise = item.get('epsSurprise', 0)
            
            # Calculate YoY changes
            revenue_yoy = calculate_yoy_change(revenue, revenue_prior)
            eps_yoy = calculate_yoy_change(eps, eps_prior)
            
            # Format numbers
            revenue_formatted = format_number(revenue)
            revenue_surprise_formatted = format_number(abs(revenue_surprise))
            
            # Determine growth/decline text
            revenue_trend = "growth" if revenue_yoy >= 0 else "decline"
            eps_trend = "growth" if eps_yoy >= 0 else "decline"
            
            # Create hyperlink for symbol
            symbol_link = f"[{symbol}](https://stocknear.com/stocks/{symbol})"
            
            # Format the entry text with nested bullet points
            entry = (
                f"**{name}** ({symbol_link}) has released its quarterly earnings at {time}:\n\n"
                f"* Revenue of {revenue_formatted} "
                f"{'exceeds' if revenue_surprise > 0 else 'misses'} estimates by {revenue_surprise_formatted}, "
                f"with {revenue_yoy:.2f}% YoY {revenue_trend}.\n\n"
                f"* EPS of ${eps:.2f} "
                f"{'exceeds' if eps_surprise > 0 else 'misses'} estimates by ${abs(eps_surprise):.2f}, "
                f"with {eps_yoy:.2f}% YoY {eps_trend}.\n\n"
            )

            formatted_items.append(entry)
    
    support_message = "\nInvest in yourself and embrace data-driven decisions to minimize losses, identify opportunities and achieve consistent growth with [Stocknear](https://stocknear.com/pricing) 🚀"
    
    return "".join(formatted_items) + support_message

def format_afterhour_market():
    try:
        # Load gainers data
        with open("json/market-movers/afterhours/gainers.json", 'r') as file:
            data = ujson.load(file)
            gainers = [
                {'symbol': item['symbol'], 'name': item['name'], 'price': item['price'], 
                 'changesPercentage': item['changesPercentage'], 'marketCap': item['marketCap']} 
                for item in data[:5]
            ]
        
        # Load losers data
        with open("json/market-movers/afterhours/losers.json", 'r') as file:
            data = ujson.load(file)
            losers = [
                {'symbol': item['symbol'], 'name': item['name'], 'price': item['price'], 
                 'changesPercentage': item['changesPercentage'], 'marketCap': item['marketCap']} 
                for item in data[:5]
            ]
        
        market_movers = {'gainers': gainers, 'losers': losers}
    
    except Exception as e:
        print(f"Error loading market data: {e}")
        market_movers = {'gainers': [], 'losers': []}
    
    # Create Gainers Table
    gainers_table = "| Symbol | Name | Price | Change (%) | Market Cap |\n"
    gainers_table += "|:------:|:-----|------:|-----------:|-----------:|\n"
    for gainer in market_movers["gainers"]:
        gainers_table += (
            f"| [{gainer['symbol']}](https://stocknear.com/stocks/{gainer['symbol']}) | {gainer['name'][:30]} | "
            f"{gainer['price']:.2f} | +{gainer['changesPercentage']:.2f}% | "
            f"{format_number(gainer['marketCap'])} |\n"
        )

    # Create Losers Table
    losers_table = "| Symbol | Name | Price | Change (%) | Market Cap |\n"
    losers_table += "|:------:|:-----|------:|-----------:|-----------:|\n"
    for loser in market_movers["losers"]:
        losers_table += (
            f"| [{loser['symbol']}](https://stocknear.com/stocks/{loser['symbol']}) | {loser['name'][:30]} | "
            f"{loser['price']:.2f} | {loser['changesPercentage']:.2f}% | "
            f"{format_number(loser['marketCap'])} |\n"
        )
    # Construct final markdown text
    return f"""

Here's a summary of today's After-Hours Gainers and Losers, showcasing stocks that stood out after the market closed.

### 📈 After-Hours Gainers

{gainers_table}

### 📉 After-Hours Losers

{losers_table}

More info can be found here: [After-Hours Gainers and Losers](https://stocknear.com/market-mover/afterhours/gainers)
"""

def format_premarket_market():
    try:
        # Load gainers data
        with open("json/market-movers/premarket/gainers.json", 'r') as file:
            data = ujson.load(file)
            gainers = [
                {'symbol': item['symbol'], 'name': item['name'], 'price': item['price'], 
                 'changesPercentage': item['changesPercentage'], 'marketCap': item['marketCap']} 
                for item in data[:5]
            ]
        
        # Load losers data
        with open("json/market-movers/premarket/losers.json", 'r') as file:
            data = ujson.load(file)
            losers = [
                {'symbol': item['symbol'], 'name': item['name'], 'price': item['price'], 
                 'changesPercentage': item['changesPercentage'], 'marketCap': item['marketCap']} 
                for item in data[:5]
            ]
        
        market_movers = {'gainers': gainers, 'losers': losers}
    
    except Exception as e:
        print(f"Error loading market data: {e}")
        market_movers = {'gainers': [], 'losers': []}
    
    # Create Gainers Table
    gainers_table = "| Symbol | Name | Price | Change (%) | Market Cap |\n"
    gainers_table += "|:------:|:-----|------:|-----------:|-----------:|\n"
    for gainer in market_movers["gainers"]:
        gainers_table += (
            f"| [{gainer['symbol']}](https://stocknear.com/stocks/{gainer['symbol']}) | {gainer['name'][:30]} | "
            f"{gainer['price']:.2f} | +{gainer['changesPercentage']:.2f}% | "
            f"{format_number(gainer['marketCap'])} |\n"
        )

    # Create Losers Table
    losers_table = "| Symbol | Name | Price | Change (%) | Market Cap |\n"
    losers_table += "|:------:|:-----|------:|-----------:|-----------:|\n"
    for loser in market_movers["losers"]:
        losers_table += (
            f"| [{loser['symbol']}](https://stocknear.com/stocks/{loser['symbol']}) | {loser['name'][:30]} | "
            f"{loser['price']:.2f} | {loser['changesPercentage']:.2f}% | "
            f"{format_number(loser['marketCap'])} |\n"
        )
    # Construct final markdown text
    return f"""

Here's a summary of today's Premarket Gainers and Losers, showcasing stocks that stood out before the market opened.

### 📈 Premarket Gainers

{gainers_table}

### 📉 Premarket Losers

{losers_table}

More info can be found here: [Premarket Gainers and Losers](https://stocknear.com/market-mover/premarket/gainers)
"""


def format_option_data():
    try:
        with open("json/stocks-list/list/highest-option-premium.json", 'r') as file:
            data = ujson.load(file)
            highest_premium = [
                {'symbol': item['symbol'], 
                 'changesPercentage': item['changesPercentage'], 'totalPrem': item['totalPrem'],'totalOI': item['totalOI'],'ivRank':item['ivRank']} 
                for item in data[:5]
            ]

        with open("json/stocks-list/list/highest-option-iv-rank.json", 'r') as file:
            data = ujson.load(file)
            highest_iv_rank = [
                {'symbol': item['symbol'], 
                 'changesPercentage': item['changesPercentage'], 'totalPrem': item['totalPrem'],'totalOI': item['totalOI'],'ivRank':item['ivRank']} 
                for item in data[:5]
            ]

        with open("json/stocks-list/list/highest-open-interest-change.json", 'r') as file:
            data = ujson.load(file)
            highest_change_oi = [
                {'symbol': item['symbol'], 
                 'changesPercentage': item['changesPercentage'], 'totalPrem': item['totalPrem'],'changeOI': item['changeOI'],'ivRank':item['ivRank']} 
                for item in data[:5]
            ]

        
        combined_data = {'highest_premium': highest_premium, 'highest_iv_rank': highest_iv_rank, 'highest_change_oi': highest_change_oi}
    
    except Exception as e:
        print(f"Error loading market data: {e}")
        combined_data = {'highest_premium': [], 'highest_iv_rank': []}
    
    # Create highest_premium Table
    highest_premium_table = "| Symbol | Change (%) | Total Prem | IV Rank | Total OI |\n"
    highest_premium_table += "|:------:|:-----|------:|-----------:|-----------:|\n"
    for item in combined_data["highest_premium"]:
        highest_premium_table += (
            f"| [{item['symbol']}](https://stocknear.com/stocks/{item['symbol']}) | {item['changesPercentage']:.2f}% | "
            f"{format_number(item['totalPrem'])} | {item['ivRank']:.2f} | "
            f"{format_number(item['totalOI'])} |\n"
        )

    # Create highest_iv_rank Table
    highest_iv_rank_table = "| Symbol | Change (%) | Total Prem | IV Rank | Total OI |\n"
    highest_iv_rank_table += "|:------:|:-----|------:|-----------:|-----------:|\n"
    for item in combined_data["highest_iv_rank"]:
        highest_iv_rank_table += (
            f"| [{item['symbol']}](https://stocknear.com/stocks/{item['symbol']}) | {item['changesPercentage']:.2f}% | "
            f"{format_number(item['totalPrem'])} | {item['ivRank']:.2f} | "
            f"{format_number(item['totalOI'])} |\n"
        )

    # Create highest_iv_rank Table
    highest_change_oi_table = "| Symbol | Change (%) | Total Prem | IV Rank | OI Change|\n"
    highest_change_oi_table += "|:------:|:-----|------:|-----------:|-----------:|\n"
    for item in combined_data["highest_change_oi"]:
        highest_change_oi_table += (
            f"| [{item['symbol']}](https://stocknear.com/stocks/{item['symbol']}) | {item['changesPercentage']:.2f}% | "
            f"{format_number(item['totalPrem'])} | {item['ivRank']:.2f} | "
            f"{format_number(item['changeOI'], decimal=True)} |\n"
        )

    # Construct final markdown text
    return f"""

Here's a quick overview of the top companies that led the market today with the highest options premium, IV rank and notable open interest (OI) changes—highlighting key stocks that gained attention.
### Highest Options Premium

{highest_premium_table}

### Top IV Rank Leaders

{highest_iv_rank_table}

### Hottest Companies with highest OI Change

{highest_change_oi_table}

More info can be found at [Stocknear](https://stocknear.com/list)
"""




def create_post(post_data):
    include_rsi = False
    include_volume = post_data['data_type'] == 'penny-stocks'

    try:
        # Use the parameter passed to the function
        with open(f"json/stocks-list/list/{post_data['data_type']}.json", 'r') as file:
            data = ujson.load(file)
        
        # Limit to first 5 items and select specific fields
        include_rsi = post_data['data_type'] in ["overbought-stocks", "oversold-stocks"]
        data = [
            {
                'rank': item['rank'],
                'symbol': item['symbol'], 
                'price': item['price'], 
                'changesPercentage': item['changesPercentage'], 
                'marketCap': item['marketCap'],
                'rsi': item.get('rsi') if include_rsi else None,
                'volume': item.get('volume') if include_volume else None
            } 
            for item in data[:5]
        ]
    
    except Exception as e:
        print(f"Error loading data: {e}")
        data = []
    
    # Create Markdown table headers
    if include_rsi:
        data_table = "| Rank | Symbol | RSI | Price | Change (%) | Market Cap |\n"
        data_table += "|:----:|:------|----:|------:|-----------:|-----------:|\n"
    elif include_volume:
        data_table = "| Rank | Symbol | Price | Change (%) | Volume | Market Cap |\n"
        data_table += "|:----:|:------|------:|-----------:|-------:|-----------:|\n"
    else:
        data_table = "| Rank | Symbol | Price | Change (%) | Market Cap |\n"
        data_table += "|:----:|:------|------:|-----------:|-----------:|\n"
    
    # Generate table rows
    for item in data:
        if include_rsi:
            data_table += (
                f"| {item['rank']} | [{item['symbol']}](https://stocknear.com/stocks/{item['symbol']}) | "
                f"{item['rsi']:.2f} | {item['price']:.2f} | {'+' if item['changesPercentage'] > 0 else ''}{item['changesPercentage']:.2f}% | "
                f"{format_number(item['marketCap'])} |\n"
            )
        elif include_volume:
            data_table += (
                f"| {item['rank']} | [{item['symbol']}](https://stocknear.com/stocks/{item['symbol']}) | "
                f"{item['price']:.2f} | {'+' if item['changesPercentage'] > 0 else ''}{item['changesPercentage']:.2f}% | "
                f"{format_number(item['volume'])} | {format_number(item['marketCap'])} |\n"
            )
        else:
            data_table += (
                f"| {item['rank']} | [{item['symbol']}](https://stocknear.com/stocks/{item['symbol']}) | "
                f"{item['price']:.2f} | {'+' if item['changesPercentage'] > 0 else ''}{item['changesPercentage']:.2f}% | "
                f"{format_number(item['marketCap'])} |\n"
            )
    
    # Return the Markdown string
    return f"""


{data_table}

The complete list can be found [here]({post_data['url']})

*{post_data['info_text']}*

*PS: If you find this post valuable please leave an upvote. Would love to hear what you guys think.*
"""




def post_to_reddit():
    # Load environment variables
    load_dotenv()
    args = parse_args()
    post_type = args.post_type
    

    # Initialize Reddit instance
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_BOT_API_KEY'),
        client_secret=os.getenv('REDDIT_BOT_API_SECRET'),
        username=os.getenv('REDDIT_USERNAME'),
        password=os.getenv('REDDIT_PASSWORD'),
        user_agent=os.getenv('REDDIT_USER_AGENT', 'script:my_bot:v1.0 (by /u/username)')
    )
    
    # Define the subreddit
    subreddit = reddit.subreddit("stocknear")

    flair_choices = subreddit.flair.link_templates  # Get submission flair templates

    # Print all submission flairs
    '''
    print("Submission Flairs:")
    for flair in flair_choices:
        print(f"ID: {flair['id']} | Text: {flair['text']} | CSS Class: {flair['css_class']} | Mod Only: {flair['mod_only']}")
    '''

    
    # Get current date with formatting
    today = datetime.now()
    month_str = today.strftime("%b")
    day = today.day
    year = today.year
    day_suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    formatted_date = f"{month_str} {day}{day_suffix} {year}"
    
    # Load and parse data from JSON file
    with open("json/dashboard/data.json", "rb") as file:
        data = orjson.loads(file.read())
    
    if post_type == 'stock-list':
        post_configs = [
            {
                "data_type": "penny-stocks",
                "title": f"Top 5 Actively Traded Penny Stocks by Volume 🚀",
                "url": "https://stocknear.com/list/penny-stocks",
                "info_text": "Penny stocks are generally defined as stocks trading below $5 per share. This list is filtered to show only stocks with a volume over 10K.",
                "flair_id": "b348676c-e451-11ee-8572-328509439585"
            },
            {
                "data_type": "overbought-stocks",
                "title": f"Top 5 Most Overbought Companies 📉",
                "url": "https://stocknear.com/list/overbought-stocks",
                'info_text': "I’ve compiled a list of the top 5 most overbought companies based on RSI (Relative Strength Index) data. For those who don’t know, RSI is a popular indicator that ranges from 0 to 100, with values above 70 typically indicating that a stock is overbought.",
                "flair_id": "b348676c-e451-11ee-8572-328509439585"
            },
            {
                "data_type": "oversold-stocks",
                "title": f"Top 5 Most Oversold Companies 📈",
                "url": "https://stocknear.com/list/oversold-stocks",
                'info_text': "I’ve compiled a list of the top 5 most oversold companies based on RSI (Relative Strength Index) data. For those who don’t know, RSI is a popular indicator that ranges from 0 to 100, with values below 30 typically indicating that a stock is oversold.",
                "flair_id": "b348676c-e451-11ee-8572-328509439585"
            },
        ]

        for item in post_configs:
            formatted_text = create_post(item)
            title = item["title"]
            flair_id = item["flair_id"]
            
            # Submit the post
            post = subreddit.submit(
                title=title,
                selftext=formatted_text,
                flair_id=flair_id
            )
        

    if post_type == 'earnings':
        # Define the post configurations
        post_configs = [
            {
                "data_key": "upcomingEarnings",
                "format_func": format_upcoming_earnings_data,
                "title": f"Upcoming Earnings for {formatted_date}",
                "flair_id": "b9f76638-772e-11ef-96c1-0afbf26bd890"
            },
            '''
            {
                "data_key": "recentEarnings",
                "format_func": format_recent_earnings_data,
                "title": f"Recent Earnings for {formatted_date}",
                "flair_id": "b9f76638-772e-11ef-96c1-0afbf26bd890"
            },
            '''
        ]
    
        try:
            # Loop through post configurations to submit each post
            for config in post_configs:
                if len(data.get(config["data_key"], [])) > 0:
                    formatted_text = config["format_func"](data.get(config["data_key"], []))
                    title = config["title"]
                    flair_id = config["flair_id"]
                    
                    # Submit the post
                    post = subreddit.submit(
                        title=title,
                        selftext=formatted_text,
                        flair_id=flair_id
                    )
                    print(f"Post created successfully")
        
        except praw.exceptions.PRAWException as e:
            print(f"Error posting to Reddit: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")

    if post_type == 'premarket':
        try:
            formatted_content = format_premarket_market()
            title = "Premarket Gainers and Losers for Today 🚀📉"
            post = subreddit.submit(title, selftext=formatted_content, flair_id="b348676c-e451-11ee-8572-328509439585")
            print(f"Post created successfully")
        except Exception as e:
            print(f"Error posting to Reddit: {str(e)}")

    if post_type == 'aftermarket':
        try:
            formatted_content = format_afterhour_market()
            title = "Afterhours Gainers and Losers for Today 🚀📉"
            post = subreddit.submit(title, selftext=formatted_content, flair_id="b348676c-e451-11ee-8572-328509439585")
            print(f"Post created successfully")
        except Exception as e:
            print(f"Error posting to Reddit: {str(e)}")
    

    if post_type == 'option':
       
        try:
            formatted_content = format_option_data()
            title = "Top Companies with the Highest Options Premiums, IV Rank and OI Change Today 🚀📉"
            post = subreddit.submit(title, selftext=formatted_content, flair_id="b348676c-e451-11ee-8572-328509439585")
            print(f"Post created successfully")
        except Exception as e:
            print(f"Error posting to Reddit: {str(e)}")



if __name__ == "__main__":
    post_to_reddit()
