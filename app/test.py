from functions import *
from functions import FUNCTION_SOURCE_METADATA
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents.stream_events import RunItemStreamEvent
from agents import Agent, Runner, ModelSettings
import json
import asyncio
from datetime import datetime

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("json/llm/chat_instruction.txt","r",encoding="utf-8") as file:
    CHAT_INSTRUCTION = file.read()

# Planning instruction for the agent
PLANNING_INSTRUCTION = """
You are a financial analyst AI. Before analyzing a stock, you must first create a detailed plan.
For each step in your analysis, provide:
1. A title in the format **Step X: Title**
2. A brief description of what will be analyzed

After planning, do not execute functions. For retrieving data use only the selected_tools.
"""

selected_tools = [search_new_listed_companies, get_news_flow, get_reddit_tracker, get_fear_and_greed_index, 
             get_ticker_earnings_call_transcripts, get_all_sector_overview, get_bitcoin_etfs, 
             get_most_shorted_stocks, get_penny_stocks, get_ipo_calendar, get_dividend_calendar, 
             get_ticker_trend_forecast, get_monthly_dividend_stocks, get_top_rated_dividend_stocks, 
             get_dividend_aristocrats, get_dividend_kings, get_overbought_tickers, get_oversold_tickers, 
             get_ticker_owner_earnings, get_ticker_financial_score, get_ticker_key_metrics, 
             get_ticker_statistics, get_ticker_dividend, get_ticker_dark_pool, get_ticker_unusual_activity, 
             get_ticker_open_interest_by_strike_and_expiry, get_ticker_max_pain, 
             get_ticker_options_overview_data, get_ticker_shareholders, get_ticker_insider_trading, 
             get_ticker_pre_post_quote, get_ticker_quote, get_market_flow, get_market_news, 
             get_analyst_tracker, get_latest_congress_trades, get_insider_tracker, get_potus_tracker, 
             get_top_active_stocks, get_top_aftermarket_losers, get_top_premarket_losers, get_top_losers, 
             get_top_aftermarket_gainers, get_top_premarket_gainers, get_top_gainers, 
             get_ticker_analyst_rating, get_ticker_news, get_latest_dark_pool_feed, 
             get_latest_options_flow_feed, get_ticker_bull_vs_bear, get_ticker_earnings, 
             get_ticker_earnings_price_reaction, get_top_rating_stocks, get_economic_calendar, 
             get_earnings_releases, get_ticker_analyst_estimate, get_ticker_business_metrics, 
             get_why_priced_moved, get_ticker_short_data, get_company_data, 
             get_ticker_hottest_options_contracts, get_ticker_ratios_statement, 
             get_ticker_cash_flow_statement, get_ticker_income_statement, 
             get_ticker_balance_sheet_statement, get_congress_activity]

user_query = "should I buy Intel right now"

# Planning Agent - First create the plan
planning_agent = Agent(
    name="Planning Agent",
    instructions=PLANNING_INSTRUCTION,
    model=os.getenv("CHAT_MODEL"),
    tools=selected_tools,  # No tools for planning phase, just thinking
)

async def create_analysis_plan(agent, query):
    """Generate the analysis plan with steps and function calls"""
    print("=" * 60)
    print("GENERATING ANALYSIS PLAN")
    print("=" * 60)
    print()

    planning_prompt = f"""
    Create a detailed step-by-step plan to analyze answer the user query {query}.
    Make it max 5. steps.

    For each step provide:
    - **Step Number: Title**
    - Description of what will be analyzed

    Format each step clearly and include the exact function names from the available tools.
    """

    # Use Runner.run_streamed to get the plan
    result = Runner.run_streamed(planning_agent, input=planning_prompt)

    full_content = ""

    async for event in result.stream_events():
        try:
            # Process only raw_response_event events
            if event.type == "raw_response_event":
                delta = getattr(event.data, "delta", "")
                if delta:
                    full_content += delta
                    print(full_content)

        except Exception as e:
            print(f"Error processing event: {e}")

   

async def analyze_with_planning():
    """Main function to run the complete analysis with planning"""
    
    await create_analysis_plan(planning_agent, user_query)
      

# Run the analysis
if __name__ == "__main__":
    asyncio.run(analyze_with_planning())