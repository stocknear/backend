"""
Enhanced Rule Extractor for Stock Screener
Uses complete frontend allRules context to build accurate screening rules
"""

import os
import orjson
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
import re

load_dotenv()
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Complete allRules definition from frontend (from the user's message)
ALL_RULES = {
    "avgVolume": {
        "label": "Average Volume",
        "step": ["100M", "10M", "1M", "100K", "10K", "1K", "0"],
        "defaultCondition": "over",
        "defaultValue": "any"
    },
    "volume": {
        "label": "Volume",
        "step": ["100M", "10M", "1M", "100K", "10K", "1K", "0"],
        "defaultCondition": "over",
        "defaultValue": "any"
    },
    "rsi": {
        "label": "Relative Strength Index",
        "step": [90, 80, 70, 60, 50, 40, 30, 20],
        "defaultCondition": "over",
        "defaultValue": "any"
    },
    "stochRSI": {
    "label": "Stochastic RSI Fast",
    "step": [90, 80, 70, 60, 50, 40, 30, 20],
        "defaultCondition": "over",
    "defaultValue": "any"
    },
    "mfi": {
    "label": "Money Flow Index",
    "step": [90, 80, 70, 60, 50, 40, 30, 20],
        "defaultCondition": "over",
    "defaultValue": "any"
    },
    "cci": {
    "label": "Commodity Channel Index",
    "step": [250, 200, 100, 50, 20, 0, -20, -50, -100, -200, -250],
        "defaultCondition": "over",
    "defaultValue": "any"
    },
    "atr": {
    "label": "Average True Range",
    "step": [20, 15, 10, 5, 3, 1],
        "defaultCondition": "over",
    "defaultValue": "any"
    },
    "sma20": {
    "label": "20-Day Moving Average",
    "step": [
        "Price above SMA20",
        "SMA20 above SMA50",
        "SMA20 above SMA100",
        "SMA20 above SMA200",
        "Price below SMA20",
        "SMA20 below SMA50",
        "SMA20 below SMA100",
        "SMA20 below SMA200",
      ],
        "defaultValue": "any"
    },
    "sma50": {
    "label": "50-Day Moving Average",
    "step": [
        "Price above SMA50",
        "SMA50 above SMA20",
        "SMA50 above SMA100",
        "SMA50 above SMA200",
        "Price below SMA50",
        "SMA50 below SMA20",
        "SMA50 below SMA100",
        "SMA50 below SMA200",
      ],
        "defaultValue": "any"
    },
    "sma100": {
    "label": "100-Day Moving Average",
    "step": [
        "Price above SMA100",
        "SMA100 above SMA20",
        "SMA100 above SMA50",
        "SMA100 above SMA200",
        "Price below SMA100",
        "SMA100 below SMA20",
        "SMA100 below SMA50",
        "SMA100 below SMA200",
      ],
        "defaultValue": "any"
    },
    "sma200": {
    "label": "200-Day Moving Average",
    "step": [
        "Price above SMA200",
        "SMA200 above SMA20",
        "SMA200 above SMA50",
        "SMA200 above SMA100",
        "Price below SMA200",
        "SMA200 below SMA20",
        "SMA200 below SMA50",
        "SMA200 below SMA100",
      ],
        "defaultValue": "any"
    },
    "ema20": {
    "label": "20-Day Exp. Moving Average",
    "step": [
        "Price above EMA20",
        "EMA20 above EMA50",
        "EMA20 above EMA100",
        "EMA20 above EMA200",
        "Price below EMA20",
        "EMA20 below EMA50",
        "EMA20 below EMA100",
        "EMA20 below EMA200",
      ],
        "defaultValue": "any"
    },
    "ema50": {
    "label": "50-Day Exp. Moving Average",
    "step": [
        "Price above EMA50",
        "EMA50 above EMA20",
        "EMA50 above EMA100",
        "EMA50 above EMA200",
        "Price below EMA50",
        "EMA50 below EMA20",
        "EMA50 below EMA100",
        "EMA50 below EMA200",
      ],
        "defaultValue": "any"
    },
    "ema100": {
    "label": "100-Day Exp. Moving Average",
    "step": [
        "Price above EMA100",
        "EMA100 above EMA20",
        "EMA100 above EMA50",
        "EMA100 above EMA200",
        "Price below EMA100",
        "EMA100 below EMA20",
        "EMA100 below EMA50",
        "EMA100 below EMA200",
      ],
        "defaultValue": "any"
    },
    "ema200": {
    "label": "200-Day Exp. Moving Average",
    "step": [
        "Price above EMA200",
        "EMA200 above EMA20",
        "EMA200 above EMA50",
        "EMA200 above EMA100",
        "Price below EMA200",
        "EMA200 below EMA20",
        "EMA200 below EMA50",
        "EMA200 below EMA100",
      ],
        "defaultValue": "any"
    },
    "grahamNumber": {
    "label": "Graham Number",
    "step": ["Price > Graham Number", "Price < Graham Number"],
    "defaultValue": "any"},
    "grahamUpside": {
    "label": "Graham Upside",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "lynchUpside": {
    "label": "Lynch Upside",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "lynchFairValue": {
    "label": "Lynch Fair Value",
    "step": ["Price > Lynch Fair Value", "Price < Lynch Fair Value"],
    "defaultValue": "any"},
    "price": {
    "label": "Price",
    "step": [1000, 500, 400, 300, 200, 150, 100, 80, 60, 50, 20, 10, 5, 1],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "changesPercentage": {
    "label": "Price Change 1D",
    "step": ["20%", "10%", "5%", "1%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "change1W": {
    "label": "Price Change 1W",
    "step": ["20%", "10%", "5%", "1%", "-1%", "-5%", "-10%", "-20%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "change1M": {
    "label": "Price Change 1M",
    "step": [
        "100%",
        "50%",
        "20%",
        "10%",
        "5%",
        "1%",
        "-1%",
        "-5%",
        "-10%",
        "-20%",
        "-50%",
      ],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "change3M": {
    "label": "Price Change 3M",
    "step": [
        "100%",
        "50%",
        "20%",
        "10%",
        "5%",
        "1%",
        "-1%",
        "-5%",
        "-10%",
        "-20%",
        "-50%",
      ],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "change6M": {
    "label": "Price Change 6M",
    "step": [
        "100%",
        "50%",
        "20%",
        "10%",
        "5%",
        "1%",
        "-1%",
        "-5%",
        "-10%",
        "-20%",
        "-50%",
      ],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "change1Y": {
    "label": "Price Change 1Y",
    "step": [
        "100%",
        "50%",
        "20%",
        "10%",
        "5%",
        "1%",
        "-1%",
        "-5%",
        "-10%",
        "-20%",
        "-50%",
      ],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "change3Y": {
    "label": "Price Change 3Y",
    "step": [
        "100%",
        "50%",
        "20%",
        "10%",
        "5%",
        "1%",
        "-1%",
        "-5%",
        "-10%",
        "-20%",
        "-50%",
      ],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "marketCap": {
    "label": "Market Cap",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "workingCapital": {
    "label": "Working Capital",
    "step": ["20B", "10B", "5B", "1B", "500M", "100M", "50M", "10M", "1M", "0"],
    "defaultCondition": "over",
    "defaultValue": "any"
    },
    "totalAssets": {
    "label": "Total Assets",
    "step": ["500B", "200B", "100B", "50B", "10B", "1B", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"
    },
    "tangibleAssetValue": {
    "label": "Tangible Assets",
    "step": ["500B", "200B", "100B", "50B", "10B", "1B", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"
    },
    "revenue": {
    "label": "Revenue",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "revenueGrowthYears": {
    "label": "Revenue Growth Years",
    "step": ["10", "5", "3", "1"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "epsGrowthYears": {
    "label": "EPS Growth Years",
    "step": ["10", "5", "3", "1"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "netIncomeGrowthYears": {
    "label": "Net Income Growth Years",
    "step": ["10", "5", "3", "1"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "grossProfitGrowthYears": {
    "label": "Gross Profit Growth Years",
    "step": ["10", "5", "3", "1"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "growthRevenue": {
    "label": "Revenue Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "costOfRevenue": {
    "label": "Cost of Revenue",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "growthCostOfRevenue": {
    "label": "Cost of Revenue Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "varType": "percentSign",
    "defaultValue": "any"},
    "costAndExpenses": {
    "label": "Cost & Expenses",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "growthCostAndExpenses": {
    "label": "Cost & Expenses Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "varType": "percentSign",
    "defaultValue": "any"},
    "netIncome": {
    "label": "Net Income",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "growthNetIncome": {
    "label": "Net Income Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "grossProfit": {
    "label": "Gross Profit",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "growthGrossProfit": {
    "label": "Gross Profit Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "researchAndDevelopmentExpenses": {
    "label": "Research & Development",
    "step": ["10B", "1B", "100M", "10M", "1M", 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "growthResearchAndDevelopmentExpenses": {
    "label": "R&D Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "varType": "percentSign",
    "defaultValue": "any"},
    "payoutRatio": {
    "label": "Payout Ratio",
    "step": ["100%", "80%", "60%", "40%", "20%", "0%", "-20%", "-40%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percent"
    },
    "dividendYield": {
    "label": "Dividend Yield",
    "step": ["50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percent"
    },
    "payoutFrequency": {
    "label": "Dividend Payout Frequency",
    "step": ["Monthly", "Quarterly", "Semi-Annual", "Annual"],
    "defaultCondition": "",
    "defaultValue": "any"},
    "annualDividend": {
    "label": "Annual Dividend",
    "step": [10, 5, 3, 2, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "dividendGrowth": {
    "label": "Dividend Growth",
    "step": ["50%", "20%", "10%", "5%", "3%", "2%", "1%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percent"
    },
    "eps": {
    "label": "EPS",
    "step": [20, 15, 10, 5, 3, 2, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "growthEPS": {
    "label": "EPS Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "interestIncome": {
    "label": "Interest Income",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"
    },
    "interestExpense": {
    "label": "Interest Expenses",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "growthInterestExpense": {
    "label": "Interest Expenses Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "varType": "percentSign",
    "defaultValue": "any"},
    "operatingExpenses": {
    "label": "Operating Expenses",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "growthOperatingExpenses": {
    "label": "Operating Expenses Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "varType": "percentSign",
    "defaultValue": "any"},
    "ebit": {
    "label": "EBIT",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "operatingIncome": {
    "label": "Operating Income",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "growthOperatingIncome": {
    "label": "Operating Income Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "growthFreeCashFlow": {
    "label": "FCF Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "growthOperatingCashFlow": {
    "label": "OCF Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "growthStockBasedCompensation": {
    "label": "SBC Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "varType": "percentSign",
    "defaultValue": "any"},
    "growthTotalLiabilities": {
    "label": "Total Liabilities Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "growthTotalDebt": {
    "label": "Total Debt Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "growthTotalStockholdersEquity": {
    "label": "Shareholders Equity Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "researchDevelopmentRevenueRatio": {
    "label": "R&D / Revenue",
    "step": ["20%", "10%", "5%", "1%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "cagr3YearRevenue": {
    "label": "Revenue CAGR 3Y",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "cagr5YearRevenue": {
    "label": "Revenue CAGR 5Y",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "cagr3YearEPS": {
    "label": "EPS CAGR 3Y",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "cagr5YearEPS": {
    "label": "EPS CAGR 5Y",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "cagrNext3YearEPS": {
    "label": "EPS Growth Next 3Y",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "cagrNext5YearEPS": {
    "label": "EPS Growth Next 5Y",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "cagrNext3YearRevenue": {
    "label": "Revenue Growth Next 3Y",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "cagrNext5YearRevenue": {
    "label": "Revenue Growth Next 5Y",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "returnOnInvestedCapital": {
    "label": "Return On Invested Capital",
    "step": ["80%", "50%", "20%", "10%", "5%", "0%"],
    "defaultCondition": "over",
    "varType": "percentSign",
    "defaultValue": "any"},
    "returnOnCapitalEmployed": {
    "label": "Return On Capital Employed",
    "step": ["100%", "50%", "20%", "10%", "5%", "0%"],
    "defaultCondition": "over",
    "varType": "percentSign",
    "defaultValue": "any"},
    "relativeVolume": {
    "label": "Relative Volume",
    "step": ["500%", "200%", "100%", "50%", "10%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "institutionalOwnership": {
    "label": "Institutional Ownership",
    "step": ["90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percent"
    },
    "priceToEarningsGrowthRatio": {
    "label": "PEG Ratio",
    "step": [100, 10, 5, 3, 1, 0.5, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "forwardPE": {
    "label": "Forward PE",
    "step": [50, 20, 10, 5, 1, 0, -1, -5, -10, -20, -50],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "forwardPS": {
    "label": "Forward PS",
    "step": [50, 20, 10, 5, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "priceToBookRatio": {
    "label": "PB Ratio",
    "step": [50, 40, 30, 20, 10, 5, 1],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "priceToSalesRatio": {
    "label": "PS Ratio",
    "step": [50, 40, 30, 20, 10, 5, 1],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "beta": {
    "label": "Beta",
    "step": [10, 5, 1, -5, -10],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "ebitda": {
    "label": "EBITDA",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "growthEBITDA": {
    "label": "EBITDA Growth",
    "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
    "defaultCondition": "over",
    "varType": "percentSign",
    "defaultValue": "any"},
    "var": {
    "label": "Value-at-Risk",
    "step": ["-1%", "-5%", "-10%", "-15%", "-20%"],
    "defaultCondition": "over",
    "defaultValue": "-5%",
    "varType": "percentSign"
    },
    "currentRatio": {
    "label": "Current Ratio",
    "step": [50, 40, 30, 20, 10, 5, 1],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "quickRatio": {
    "label": "Quick Ratio",
    "step": [50, 40, 30, 20, 10, 5, 1],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "debtToEquityRatio": {
    "label": "Debt / Equity",
    "step": [50, 40, 30, 20, 10, 5, 1],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "inventoryTurnover": {
    "label": "Inventory Turnover",
    "step": [200, 100, 50, 20, 10, 5, 3, 2, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"
    },
    "returnOnAssets": {
    "label": "Return on Assets",
    "step": ["80%", "50%", "20%", "10%", "5%"],
    "defaultCondition": "over",
    "varType": "percentSign",
    "defaultValue": "any"},
    "returnOnEquity": {
    "label": "Return on Equity",
    "step": ["80%", "50%", "20%", "10%", "5%"],
    "defaultCondition": "over",
    "varType": "percentSign",
    "defaultValue": "any"},
    "returnOnTangibleAssets": {
    "label": "Return on Tangible Assets",
    "step": ["80%", "50%", "20%", "10%", "5%"],
    "defaultCondition": "over",
    "varType": "percentSign",
    "defaultValue": "any"},
    "enterpriseValue": {
    "label": "Enterprise Value",
    "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "evToSales": {
    "label": "EV / Sales",
    "step": [50, 20, 10, 5, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "evToEBITDA": {
    "label": "EV / EBITDA",
    "step": [50, 20, 10, 5, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "evToEBIT": {
    "label": "EV / EBIT",
    "step": [50, 20, 10, 5, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "evToFreeCashFlow": {
    "label": "EV / FCF",
    "step": [50, 20, 10, 5, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "freeCashFlowPerShare": {
    "label": "FCF / Share",
    "step": [10, 8, 6, 4, 2, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "cashPerShare": {
    "label": "Cash / Share",
    "step": [50, 20, 10, 5, 1, 0, -1, -5, -10, -20, -50],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "priceToFreeCashFlowRatio": {
    "label": "Price / FCF",
    "step": [50, 20, 10, 5, 1, 0, -1, -5, -10, -20, -50],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "interestCoverageRatio": {
    "label": "Interest Coverage",
    "step": [10, 5, 3, 2, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "sharesShort": {
    "label": "Short Interest",
    "step": ["50M", "20M", "10M", "5M", "1M", "500K"],
    "defaultCondition": "over",
    "defaultValue": "500K"},
    "shortRatio": {
    "label": "Short Ratio",
    "step": [10, 5, 3, 2, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "shortFloatPercent": {
    "label": "Short % Float",
    "step": ["50%", "30%", "20%", "10%", "5%", "1%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percent"
    },
    "shortOutstandingPercent": {
    "label": "Short % Outstanding",
    "step": ["50%", "30%", "20%", "10%", "5%", "1%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percent"
    },
    "failToDeliver": {
    "label": "Fail to Deliver (FTD)",
    "step": ["1M", "500K", "200K", "100K", "50K", "10K", "1K"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "relativeFTD": {
    "label": "FTD / Avg. Volume",
    "step": ["300%", "200%", "100%", "50%", "20%", "10%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percent"
    },
    "freeCashFlow": {
    "label": "Free Cash Flow",
    "step": ["50B", "10B", "1B", "100M", "10M", "1M", 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "operatingCashFlow": {
    "label": "Operating Cash Flow",
    "step": ["50B", "10B", "1B", "100M", "10M", "1M", 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "operatingCashFlowPerShare": {
    "label": "Operating Cash Flow / Share",
    "step": [50, 40, 30, 10, 5, 1],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "revenuePerShare": {
    "label": "Revenue / Share",
    "step": [50, 40, 30, 10, 5, 1],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "netIncomePerShare": {
    "label": "Net Income / Share",
    "step": [50, 40, 30, 10, 5, 1],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "shareholdersEquityPerShare": {
    "label": "Shareholders Equity / Share",
    "step": [50, 40, 30, 10, 5, 1],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "interestDebtPerShare": {
    "label": "Interest Debt / Share",
    "step": [50, 40, 30, 10, 5, 1],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "capexPerShare": {
    "label": "CapEx / Share",
    "step": [50, 40, 30, 10, 5, 1],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "freeCashFlowMargin": {
    "label": "FCF Margin",
    "step": ["80%", "50%", "20%", "10%", "5%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "totalDebt": {
    "label": "Total Debt",
    "step": ["200B", "100B", "50B", "10B", "1B", "100M", "10M", "1M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "operatingCashFlowSalesRatio": {
    "label": "Operating Cash Flow / Sales",
    "step": [5, 3, 1, 0.5, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "priceToOperatingCashFlowRatio": {
    "label": "Price / Cash Flow",
    "step": [20, 15, 10, 5, 3, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "priceToEarningsRatio": {
    "label": "PE Ratio",
    "step": [100, 50, 20, 10, 5, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "stockBasedCompensation": {
    "label": "Stock-Based Compensation",
    "step": ["10B", "1B", "100M", "10M", "1M", 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "stockBasedCompensationToRevenue": {
    "label": "SBC / Revenue",
    "step": ["20%", "10%", "5%", "1%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percent"
    },
    "totalStockholdersEquity": {
    "label": "Shareholders Equity",
    "step": ["100B", "50B", "10B", "1B", "100M", "50M", "10M", "1M", 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "sharesQoQ": {
    "label": "Shares Change (QoQ)",
    "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "sharesYoY": {
    "label": "Shares Change (YoY)",
    "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "grossProfitMargin": {
    "label": "Gross Margin",
    "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "netProfitMargin": {
    "label": "Profit Margin",
    "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "pretaxProfitMargin": {
    "label": "Pretax Margin",
    "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "ebitdaMargin": {
    "label": "EBITDA Margin",
    "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "ebitMargin": {
    "label": "EBIT Margin",
    "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "operatingMargin": {
    "label": "Operating Margin",
    "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "interestIncomeToCapitalization": {
    "label": "Interest Income / Market Cap",
    "step": ["80%", "60%", "50%", "30%", "20%", "10%", "5%", "1%", "0.5%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percent"
    },
    "assetTurnover": {
    "label": "Asset Turnover",
    "step": [5, 3, 2, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"
    },
    "earningsYield": {
    "label": "Earnings Yield",
    "step": ["20%", "15%", "10%", "5%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percent"
    },
    "freeCashFlowYield": {
    "label": "FCF Yield",
    "step": ["20%", "15%", "10%", "5%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percent"
    },
    "effectiveTaxRate": {
    "label": "Effective Tax Rate",
    "step": ["20%", "15%", "10%", "5%", "0%"],
    "defaultCondition": "over",
    "varType": "percent",
    "defaultValue": "any"},
    "fixedAssetTurnover": {
    "label": "Fixed Asset Turnover",
    "step": [10, 5, 3, 2, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"
    },
    "sharesOutStanding": {
    "label": "Shares Outstanding",
    "step": ["10B", "5B", "1B", "100M", "50M", "10M", "1M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "employees": {
    "label": "Employees",
    "step": ["500K", "300K", "200K", "100K", "10K", "1K", "100"],
    "defaultCondition": "over",
    "defaultValue": "100K"},
    "revenuePerEmployee": {
    "label": "Revenue Per Employee",
    "step": ["5M", "3M", "2M", "1M", "500K", "100K", 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "profitPerEmployee": {
    "label": "Profit Per Employee",
    "step": ["5M", "3M", "2M", "1M", "500K", "100K", 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "totalLiabilities": {
    "label": "Total Liabilities",
    "step": ["500B", "200B", "100B", "50B", "10B", "1B", "100M", "10M", "1M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "altmanZScore": {
    "label": "Altman-Z-Score",
    "step": [10, 5, 3, 1],
    "defaultCondition": "over",
    "defaultValue": "any"
    },
    "piotroskiScore": {
    "label": "Piotroski F-Score",
    "step": [9, 8, 7, 6, 5, 4, 3, 2, 1],
    "defaultCondition": "over",
    "defaultValue": "any"
    },
    "earningsTime": {
    "label": "Earnings Time",
    "step": ["Before Market Open", "After Market Close"],
    "defaultCondition": "",
    "defaultValue": "any"},
    "earningsDate": {
    "label": "Earnings Date",
    "step": [
        "Today",
        "Tomorrow",
        "Next 7D",
        "Next 30D",
        "This Month",
        "Next Month",
      ],
    "defaultCondition": "",
    "defaultValue": "any",
    "varType": "date"},
    "earningsRevenueEst": {
    "label": "Earnings Revenue Estimate",
    "step": ["100B", "50B", "10B", "1B", "100M", "10M"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "earningsEPSEst": {
    "label": "Earnings EPS Estimate",
    "step": ["10", "5", "3", "2", "1", "0"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "earningsRevenueGrowthEst": {
    "label": "Revenue Estimated Growth",
    "step": ["100%", "50%", "20%", "10%", "5%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "earningsEPSGrowthEst": {
    "label": "EPS Estimated Growth",
    "step": ["100%", "50%", "20%", "10%", "5%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "analystRating": {
    "label": "Analyst Rating",
    "step": ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
    "defaultCondition": "",
    "defaultValue": "any"},
    "analystCounter": {
    "label": "Analyst Count",
    "step": ["40", "30", "20", "10", "5", "0"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "priceTarget": {
    "label": "Price Target",
    "step": ["1000", "500", "100", "10", "5", "1"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "upside": {
    "label": "Price Target Upside",
    "step": ["100%", "50%", "20%", "10%", "5%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "topAnalystRating": {
    "label": "Top Analyst Rating",
    "step": ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
    "defaultCondition": "",
    "defaultValue": "any"},
    "topAnalystCounter": {
    "label": "Top Analyst Count",
    "step": ["10", "5", "3", "1"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "topAnalystUpside": {
    "label": "Top Analyst Price Target Upside",
    "step": ["100%", "50%", "20%", "10%", "5%", "0%"],
    "defaultCondition": "over",
    "defaultValue": "any",
    "varType": "percentSign"
    },
    "topAnalystPriceTarget": {
    "label": "Top Analyst Price Target",
    "step": ["1000", "500", "100", "10", "5", "1"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    
    "score": {
    "label": "AI Score",
    "step": ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
    "defaultCondition": "",
    "defaultValue": "any"},
    "sector": {
        "label": "Sector",
        "step": [],  # sectorList placeholder
        "defaultCondition": "",
        "defaultValue": "any"
    },
    "industry": {
        "label": "Industry", 
        "step": [],  # industryList placeholder
        "defaultCondition": "",
        "defaultValue": "any"
    },
    "country": {
        "label": "Country",
        "step": [],  # listOfRelevantCountries placeholder
        "defaultCondition": "",
        "defaultValue": "any"
    },
    "ivRank": {
    "label": "IV Rank",
    "step": [50, 30, 20, 10, 5, 1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "iv30d": {
    "label": "IV 30d",
    "step": [1, 0.5, 0.3, 0.1, 0],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "totalOI": {
    "label": "Total OI",
    "step": ["500K", "300K", "200K", "100K", "50K", "10K", "1K"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "changeOI": {
    "label": "Change OI",
    "step": ["5K", "3K", "1K", "500", "300", "100"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "callVolume": {
    "label": "Call Volume",
    "step": ["100K", "50K", "20K", "10K", "5K", "1K"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "putVolume": {
    "label": "Put Volume",
    "step": ["100K", "50K", "20K", "10K", "5K", "1K"],
    "defaultCondition": "over",
    "defaultValue": "any"},
    "pcRatio": {
    "label": "P/C Ratio",
    "step": [10, 5, 3, 2, 1, 0.5],
    "defaultCondition": "over",
    "defaultValue": "any"}
}



RULE_EXTRACTION_FUNCTION = {
    "name": "extract_stock_screener_rules",
    "description": "Extract stock screening rules from natural language queries using the available frontend rules",
    "parameters": {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The rule name/field to filter on (must match available rules)"
                        },
                        "condition": {
                            "type": "string",
                            "enum": ["over", "under", "exactly", "between"],
                            "description": "The comparison condition"
                        },
                        "value": {
                            "description": "The value to compare against"
                        }
                    },
                    "required": ["name", "condition", "value"]
                }
            },
            "sort_by": {
                "type": "string",
                "description": "Optional field to sort results by"
            },
            "sort_order": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": "Sort order"
            }
        },
        "required": ["rules"]
    }
}

def build_rules_context() -> str:
    """Build a concise context of available rules for the LLM"""
    
    categories = {}
    for rule_name, rule_data in ALL_RULES.items():
        category = rule_data.get('category', 'Other')
        if isinstance(category, list):
            category = category[0]
        
        if category not in categories:
            categories[category] = []
        
        step_examples = rule_data.get('step', [])
        if len(step_examples) > 3:
            step_examples = step_examples[:3] + ["..."]
        
        categories[category].append({
            'name': rule_name,
            'label': rule_data.get('label'),
            'examples': step_examples,
            'condition': rule_data.get('defaultCondition', 'over')
        })
    
    context = "Available Stock Screening Rules:\n\n"
    
    # Prioritize most important categories
    priority_categories = [
        "Most Popular", "Price & Volume", "Short Selling Statistics", 
        "Company Info", "Performance", "Technical Analysis", "Valuation & Ratios"
    ]
    
    for category in priority_categories:
        if category in categories:
            context += f"**{category}:**\n"
            for rule in categories[category][:5]:  # Limit to 5 rules per category
                context += f"- {rule['name']} ({rule['label']}): {rule['examples']}\n"
            context += "\n"
    
    context += """
Common Query Patterns:
- "most shorted stocks" → shortFloatPercent > 20%, shortRatio > 1
- "price below $10" → price < 10  
- "large cap tech" → marketCap > 10B, sector = Technology
- "high dividend yield" → dividendYield > 3%
- "penny stocks" → price < 5

Conditions: "over", "under", "exactly", "between"
Values: Use numbers for amounts (e.g., 10 for $10), strings with units (e.g., "10B" for billions), percentages as numbers (e.g., 20 for 20%)
"""
    
    return context

async def extract_screener_rules(query: str) -> Dict[str, Any]:
    try:
        rules_context = build_rules_context()
        
        system_prompt = f"""You are a stock screener expert. Extract screening rules from user queries.

{rules_context}

Extract specific rules that match the user's request. For "most shorted stocks", use shortFloatPercent and shortRatio rules. Always include reasonable minimum market cap for liquidity."""

        response = await async_client.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            tools=[{"type": "function", "function": RULE_EXTRACTION_FUNCTION}],
            tool_choice={"type": "function", "function": {"name": "extract_stock_screener_rules"}}
        )
        
        if response.choices[0].message.tool_calls:
            args = response.choices[0].message.tool_calls[0].function.arguments
            return orjson.loads(args)
        
    except Exception as e:
        print(f"LLM extraction error: {e}")
    
    # Fallback to pattern matching
    rules = []

    price_match = re.search(r'(?:below|under)\s+(?:price\s+of\s+)?\$?(\d+(?:\.\d+)?)', query_lower)
    if price_match:
        rules.append({
            "name": "price", 
            "condition": "under",
            "value": float(price_match.group(1))
        })
        
    price_match = re.search(r'(?:above|over)\s+(?:price\s+of\s+)?\$?(\d+(?:\.\d+)?)', query_lower)
    if price_match:
        rules.append({
            "name": "price",
            "condition": "over", 
            "value": float(price_match.group(1))
        })
    
    return {"rules": rules}

async def format_rules_for_screener(extracted_rules: Dict[str, Any]) -> List[Dict]:
    """Convert extracted rules to screener format"""
    formatted_rules = []
    
    for rule in extracted_rules.get('rules', []):
        formatted_rule = {
            'name': rule.get('name'),
            'condition': rule.get('condition', 'over'),
            'value': rule.get('value')
        }
        
        # Validate rule exists in our definitions
        if rule.get('name') in ALL_RULES:
            formatted_rules.append(formatted_rule)
        
    return formatted_rules