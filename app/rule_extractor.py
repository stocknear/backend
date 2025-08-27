import os
import orjson
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
import re

load_dotenv()
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

listOfRelevantCountries = [
  "Afghanistan",
  "Albania",
  "Algeria",
  "Andorra",
  "Angola",
  "Antigua and Barbuda",
  "Argentina",
  "Armenia",
  "Australia",
  "Austria",
  "Azerbaijan",
  "Bahamas",
  "Bahrain",
  "Bangladesh",
  "Barbados",
  "Belarus",
  "Belgium",
  "Belize",
  "Benin",
  "Bhutan",
  "Bolivia",
  "Bosnia and Herzegovina",
  "Botswana",
  "Brazil",
  "Brunei",
  "Bulgaria",
  "Burkina Faso",
  "Burundi",
  "Cabo Verde",
  "Cambodia",
  "Cameroon",
  "Canada",
  "Central African Republic",
  "Chad",
  "Chile",
  "China",
  "Colombia",
  "Comoros",
  "Congo (Congo-Brazzaville)",
  "Costa Rica",
  "Croatia",
  "Cuba",
  "Cyprus",
  "Czechia (Czech Republic)",
  "Democratic Republic of the Congo",
  "Denmark",
  "Djibouti",
  "Dominica",
  "Dominican Republic",
  "Ecuador",
  "Egypt",
  "El Salvador",
  "Equatorial Guinea",
  "Eritrea",
  "Estonia",
  "Eswatini (fmr. 'Swaziland')",
  "Ethiopia",
  "Fiji",
  "Finland",
  "France",
  "Gabon",
  "Gambia",
  "Georgia",
  "Germany",
  "Ghana",
  "Greece",
  "Grenada",
  "Guatemala",
  "Guinea",
  "Guinea-Bissau",
  "Guyana",
  "Haiti",
  "Holy See",
  "Honduras",
  "Hong Kong",
  "Hungary",
  "Iceland",
  "India",
  "Indonesia",
  "Iran",
  "Iraq",
  "Ireland",
  "Israel",
  "Italy",
  "Jamaica",
  "Japan",
  "Jordan",
  "Kazakhstan",
  "Kenya",
  "Kiribati",
  "Kuwait",
  "Kyrgyzstan",
  "Laos",
  "Latvia",
  "Lebanon",
  "Lesotho",
  "Liberia",
  "Libya",
  "Liechtenstein",
  "Lithuania",
  "Luxembourg",
  "Madagascar",
  "Malawi",
  "Malaysia",
  "Maldives",
  "Mali",
  "Malta",
  "Marshall Islands",
  "Mauritania",
  "Mauritius",
  "Mexico",
  "Micronesia",
  "Moldova",
  "Monaco",
  "Mongolia",
  "Montenegro",
  "Morocco",
  "Mozambique",
  "Myanmar (formerly Burma)",
  "Namibia",
  "Nauru",
  "Nepal",
  "Netherlands",
  "New Zealand",
  "Nicaragua",
  "Niger",
  "Nigeria",
  "North Korea",
  "North Macedonia",
  "Norway",
  "Oman",
  "Pakistan",
  "Palau",
  "Palestine",
  "Panama",
  "Papua New Guinea",
  "Paraguay",
  "Peru",
  "Philippines",
  "Poland",
  "Portugal",
  "Qatar",
  "Romania",
  "Russia",
  "Rwanda",
  "Saint Kitts and Nevis",
  "Saint Lucia",
  "Saint Vincent and the Grenadines",
  "Samoa",
  "San Marino",
  "Sao Tome and Principe",
  "Saudi Arabia",
  "Senegal",
  "Serbia",
  "Seychelles",
  "Sierra Leone",
  "Singapore",
  "Slovakia",
  "Slovenia",
  "Solomon Islands",
  "Somalia",
  "South Africa",
  "South Korea",
  "South Sudan",
  "Spain",
  "Sri Lanka",
  "Sudan",
  "Suriname",
  "Sweden",
  "Switzerland",
  "Syria",
  "Tajikistan",
  "Tanzania",
  "Taiwan",
  "Thailand",
  "Timor-Leste",
  "Togo",
  "Tonga",
  "Trinidad and Tobago",
  "Tunisia",
  "Turkey",
  "Turkmenistan",
  "Tuvalu",
  "Uganda",
  "Ukraine",
  "United Arab Emirates",
  "UK",
  "United States",
  "Uruguay",
  "Uzbekistan",
  "Vanuatu",
  "Venezuela",
  "Vietnam",
  "Yemen",
  "Zambia",
  "Zimbabwe",
]

industryList = [
  "Steel",
  "Silver",
  "Other Precious Metals",
  "Gold",
  "Copper",
  "Aluminum",
  "Paper, Lumber & Forest Products",
  "Industrial Materials",
  "Construction Materials",
  "Chemicals - Specialty",
  "Chemicals",
  "Agricultural Inputs",
  "Telecommunications Services",
  "Internet Content & Information",
  "Publishing",
  "Broadcasting",
  "Advertising Agencies",
  "Entertainment",
  "Travel Lodging",
  "Travel Services",
  "Specialty Retail",
  "Luxury Goods",
  "Home Improvement",
  "Residential Construction",
  "Department Stores",
  "Personal Products & Services",
  "Leisure",
  "Gambling, Resorts & Casinos",
  "Furnishings, Fixtures & Appliances",
  "Restaurants",
  "Auto - Parts",
  "Auto - Manufacturers",
  "Auto - Recreational Vehicles",
  "Auto - Dealerships",
  "Apparel - Retail",
  "Apparel - Manufacturers",
  "Apparel - Footwear & Accessories",
  "Packaging & Containers",
  "Tobacco",
  "Grocery Stores",
  "Discount Stores",
  "Household & Personal Products",
  "Packaged Foods",
  "Food Distribution",
  "Food Confectioners",
  "Agricultural Farm Products",
  "Education & Training Services",
  "Beverages - Wineries & Distilleries",
  "Beverages - Non-Alcoholic",
  "Beverages - Alcoholic",
  "Uranium",
  "Solar",
  "Oil & Gas Refining & Marketing",
  "Oil & Gas Midstream",
  "Oil & Gas Integrated",
  "Oil & Gas Exploration & Production",
  "Oil & Gas Equipment & Services",
  "Oil & Gas Energy",
  "Oil & Gas Drilling",
  "Coal",
  "Shell Companies",
  "Investment - Banking & Investment Services",
  "Insurance - Specialty",
  "Insurance - Reinsurance",
  "Insurance - Property & Casualty",
  "Insurance - Life",
  "Insurance - Diversified",
  "Insurance - Brokers",
  "Financial - Mortgages",
  "Financial - Diversified",
  "Financial - Data & Stock Exchanges",
  "Financial - Credit Services",
  "Financial - Conglomerates",
  "Financial - Capital Markets",
  "Banks - Regional",
  "Banks - Diversified",
  "Banks",
  "Asset Management",
  "Asset Management - Bonds",
  "Asset Management - Income",
  "Asset Management - Leveraged",
  "Asset Management - Cryptocurrency",
  "Asset Management - Global",
  "Medical - Specialties",
  "Medical - Pharmaceuticals",
  "Medical - Instruments & Supplies",
  "Medical - Healthcare Plans",
  "Medical - Healthcare Information Services",
  "Medical - Equipment & Services",
  "Medical - Distribution",
  "Medical - Diagnostics & Research",
  "Medical - Devices",
  "Medical - Care Facilities",
  "Drug Manufacturers - Specialty & Generic",
  "Drug Manufacturers - General",
  "Biotechnology",
  "Waste Management",
  "Trucking",
  "Railroads",
  "Aerospace & Defense",
  "Marine Shipping",
  "Integrated Freight & Logistics",
  "Airlines, Airports & Air Services",
  "General Transportation",
  "Manufacturing - Tools & Accessories",
  "Manufacturing - Textiles",
  "Manufacturing - Miscellaneous",
  "Manufacturing - Metal Fabrication",
  "Industrial - Distribution",
  "Industrial - Specialties",
  "Industrial - Pollution & Treatment Controls",
  "Environmental Services",
  "Industrial - Machinery",
  "Industrial - Infrastructure Operations",
  "Industrial - Capital Goods",
  "Consulting Services",
  "Business Equipment & Supplies",
  "Staffing & Employment Services",
  "Rental & Leasing Services",
  "Engineering & Construction",
  "Security & Protection Services",
  "Specialty Business Services",
  "Construction",
  "Conglomerates",
  "Electrical Equipment & Parts",
  "Agricultural - Machinery",
  "Agricultural - Commodities/Milling",
  "REIT - Specialty",
  "REIT - Retail",
  "REIT - Residential",
  "REIT - Office",
  "REIT - Mortgage",
  "REIT - Industrial",
  "REIT - Hotel & Motel",
  "REIT - Healthcare Facilities",
  "REIT - Diversified",
  "Real Estate - Services",
  "Real Estate - Diversified",
  "Real Estate - Development",
  "Real Estate - General",
  "Information Technology Services",
  "Hardware, Equipment & Parts",
  "Computer Hardware",
  "Electronic Gaming & Multimedia",
  "Software - Services",
  "Software - Infrastructure",
  "Software - Application",
  "Semiconductors",
  "Media & Entertainment",
  "Communication Equipment",
  "Technology Distributors",
  "Consumer Electronics",
  "Renewable Utilities",
  "Regulated Water",
  "Regulated Gas",
  "Regulated Electric",
  "Independent Power Producers",
  "Diversified Utilities",
  "General Utilities",
]
SECTOR_LIST = [
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
]
ALL_RULES = {
    "avgVolume": {
        "label": "Average Volume",
        "step": ["100M", "10M", "1M", "100K", "10K", "1K", "0"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "volume": {
        "label": "Volume",
        "step": ["100M", "10M", "1M", "100K", "10K", "1K", "0"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "rsi": {
        "label": "Relative Strength Index",
        "step": [90, 80, 70, 60, 50, 40, 30, 20],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "stochRSI": {
        "label": "Stochastic RSI Fast",
        "step": [90, 80, 70, 60, 50, 40, 30, 20],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "mfi": {
        "label": "Money Flow Index",
        "step": [90, 80, 70, 60, 50, 40, 30, 20],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "cci": {
        "label": "Commodity Channel Index",
        "step": [250, 200, 100, 50, 20, 0, -20, -50, -100, -200, -250],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "atr": {
        "label": "Average True Range",
        "step": [20, 15, 10, 5, 3, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
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
        "defaultValue": "any",
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
        "defaultValue": "any",
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
        "defaultValue": "any",
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
        "defaultValue": "any",
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
        "defaultValue": "any",
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
        "defaultValue": "any",
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
        "defaultValue": "any",
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
        "defaultValue": "any",
    },
    "grahamNumber": {
        "label": "Graham Number",
        "step": ["Price > Graham Number", "Price < Graham Number"],
        "defaultValue": "any",
    },
    "grahamUpside": {
        "label": "Graham Upside",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
        "varType": "percentSign",
    },
    "lynchUpside": {
        "label": "Lynch Upside",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
        "varType": "percentSign",
    },
    "lynchFairValue": {
        "label": "Lynch Fair Value",
        "step": ["Price > Lynch Fair Value", "Price < Lynch Fair Value"],
        "defaultValue": "any",
    },
    "price": {
        "label": "Price",
        "step": [1000, 500, 400, 300, 200, 150, 100, 80, 60, 50, 20, 10, 5, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "changesPercentage": {
        "label": "Price Change 1D",
        "step": ["20%", "10%", "5%", "1%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "change1W": {
        "label": "Price Change 1W",
        "step": ["20%", "10%", "5%", "1%", "-1%", "-5%", "-10%", "-20%"],
        "defaultCondition": "over",
        "defaultValue": "any",
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
    },
    "marketCap": {
        "label": "Market Cap",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "workingCapital": {
        "label": "Working Capital",
        "step": ["20B", "10B", "5B", "1B", "500M", "100M", "50M", "10M", "1M", "0"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "totalAssets": {
        "label": "Total Assets",
        "step": ["500B", "200B", "100B", "50B", "10B", "1B", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "tangibleAssetValue": {
        "label": "Tangible Assets",
        "step": ["500B", "200B", "100B", "50B", "10B", "1B", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "revenue": {
        "label": "Revenue",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "revenueGrowthYears": {
        "label": "Revenue Growth Years",
        "step": ["10", "5", "3", "1"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "epsGrowthYears": {
        "label": "EPS Growth Years",
        "step": ["10", "5", "3", "1"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "netIncomeGrowthYears": {
        "label": "Net Income Growth Years",
        "step": ["10", "5", "3", "1"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "grossProfitGrowthYears": {
        "label": "Gross Profit Growth Years",
        "step": ["10", "5", "3", "1"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthRevenue": {
        "label": "Revenue Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "costOfRevenue": {
        "label": "Cost of Revenue",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthCostOfRevenue": {
        "label": "Cost of Revenue Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "costAndExpenses": {
        "label": "Cost & Expenses",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthCostAndExpenses": {
        "label": "Cost & Expenses Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "netIncome": {
        "label": "Net Income",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthNetIncome": {
        "label": "Net Income Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "grossProfit": {
        "label": "Gross Profit",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthGrossProfit": {
        "label": "Gross Profit Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "researchAndDevelopmentExpenses": {
        "label": "Research & Development",
        "step": ["10B", "1B", "100M", "10M", "1M", 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthResearchAndDevelopmentExpenses": {
        "label": "R&D Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "payoutRatio": {
        "label": "Payout Ratio",
        "step": ["100%", "80%", "60%", "40%", "20%", "0%", "-20%", "-40%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "dividendYield": {
        "label": "Dividend Yield",
        "step": ["50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "payoutFrequency": {
        "label": "Dividend Payout Frequency",
        "step": ["Monthly", "Quarterly", "Semi-Annual", "Annual"],
        "defaultCondition": "",
        "defaultValue": "any",
    },
    "annualDividend": {
        "label": "Annual Dividend",
        "step": [10, 5, 3, 2, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "dividendGrowth": {
        "label": "Dividend Growth",
        "step": ["50%", "20%", "10%", "5%", "3%", "2%", "1%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "eps": {
        "label": "EPS",
        "step": [20, 15, 10, 5, 3, 2, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthEPS": {
        "label": "EPS Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "interestIncome": {
        "label": "Interest Income",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "interestExpense": {
        "label": "Interest Expenses",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthInterestExpense": {
        "label": "Interest Expenses Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "operatingExpenses": {
        "label": "Operating Expenses",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthOperatingExpenses": {
        "label": "Operating Expenses Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "ebit": {
        "label": "EBIT",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "operatingIncome": {
        "label": "Operating Income",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthOperatingIncome": {
        "label": "Operating Income Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthFreeCashFlow": {
        "label": "FCF Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthOperatingCashFlow": {
        "label": "OCF Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthStockBasedCompensation": {
        "label": "SBC Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthTotalLiabilities": {
        "label": "Total Liabilities Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthTotalDebt": {
        "label": "Total Debt Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthTotalStockholdersEquity": {
        "label": "Shareholders Equity Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "researchDevelopmentRevenueRatio": {
        "label": "R&D / Revenue",
        "step": ["20%", "10%", "5%", "1%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "cagr3YearRevenue": {
        "label": "Revenue CAGR 3Y",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "cagr5YearRevenue": {
        "label": "Revenue CAGR 5Y",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "cagr3YearEPS": {
        "label": "EPS CAGR 3Y",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "cagr5YearEPS": {
        "label": "EPS CAGR 5Y",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "cagrNext3YearEPS": {
        "label": "EPS Growth Next 3Y",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "cagrNext5YearEPS": {
        "label": "EPS Growth Next 5Y",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "cagrNext3YearRevenue": {
        "label": "Revenue Growth Next 3Y",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "cagrNext5YearRevenue": {
        "label": "Revenue Growth Next 5Y",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "returnOnInvestedCapital": {
        "label": "Return On Invested Capital",
        "step": ["80%", "50%", "20%", "10%", "5%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "returnOnCapitalEmployed": {
        "label": "Return On Capital Employed",
        "step": ["100%", "50%", "20%", "10%", "5%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "relativeVolume": {
        "label": "Relative Volume",
        "step": ["500%", "200%", "100%", "50%", "10%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "institutionalOwnership": {
        "label": "Institutional Ownership",
        "step": ["90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "priceToEarningsGrowthRatio": {
        "label": "PEG Ratio",
        "step": [100, 10, 5, 3, 1, 0.5, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "forwardPE": {
        "label": "Forward PE",
        "step": [50, 20, 10, 5, 1, 0, -1, -5, -10, -20, -50],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "forwardPS": {
        "label": "Forward PS",
        "step": [50, 20, 10, 5, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "priceToBookRatio": {
        "label": "PB Ratio",
        "step": [50, 40, 30, 20, 10, 5, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "priceToSalesRatio": {
        "label": "PS Ratio",
        "step": [50, 40, 30, 20, 10, 5, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "beta": {
        "label": "Beta",
        "step": [10, 5, 1, -5, -10],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "ebitda": {
        "label": "EBITDA",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "growthEBITDA": {
        "label": "EBITDA Growth",
        "step": ["200%", "100%", "50%", "20%", "10%", "5%", "1%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "var": {
        "label": "Value-at-Risk",
        "step": ["-1%", "-5%", "-10%", "-15%", "-20%"],
        "defaultCondition": "over",
        "defaultValue": "-5%",
    },
    "currentRatio": {
        "label": "Current Ratio",
        "step": [50, 40, 30, 20, 10, 5, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "quickRatio": {
        "label": "Quick Ratio",
        "step": [50, 40, 30, 20, 10, 5, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "debtToEquityRatio": {
        "label": "Debt / Equity",
        "step": [50, 40, 30, 20, 10, 5, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "inventoryTurnover": {
        "label": "Inventory Turnover",
        "step": [200, 100, 50, 20, 10, 5, 3, 2, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "returnOnAssets": {
        "label": "Return on Assets",
        "step": ["80%", "50%", "20%", "10%", "5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "returnOnEquity": {
        "label": "Return on Equity",
        "step": ["80%", "50%", "20%", "10%", "5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "returnOnTangibleAssets": {
        "label": "Return on Tangible Assets",
        "step": ["80%", "50%", "20%", "10%", "5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "enterpriseValue": {
        "label": "Enterprise Value",
        "step": ["100B", "50B", "10B", "1B", "300M", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "evToSales": {
        "label": "EV / Sales",
        "step": [50, 20, 10, 5, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "evToEBITDA": {
        "label": "EV / EBITDA",
        "step": [50, 20, 10, 5, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "evToEBIT": {
        "label": "EV / EBIT",
        "step": [50, 20, 10, 5, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "evToFreeCashFlow": {
        "label": "EV / FCF",
        "step": [50, 20, 10, 5, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "freeCashFlowPerShare": {
        "label": "FCF / Share",
        "step": [10, 8, 6, 4, 2, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "cashPerShare": {
        "label": "Cash / Share",
        "step": [50, 20, 10, 5, 1, 0, -1, -5, -10, -20, -50],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "priceToFreeCashFlowRatio": {
        "label": "Price / FCF",
        "step": [50, 20, 10, 5, 1, 0, -1, -5, -10, -20, -50],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "interestCoverageRatio": {
        "label": "Interest Coverage",
        "step": [10, 5, 3, 2, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "sharesShort": {
        "label": "Short Interest",
        "step": ["50M", "20M", "10M", "5M", "1M", "500K"],
        "defaultCondition": "over",
        "defaultValue": "500K",
    },
    "shortRatio": {
        "label": "Short Ratio",
        "step": [10, 5, 3, 2, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "shortFloatPercent": {
        "label": "Short % Float",
        "step": ["50%", "30%", "20%", "10%", "5%", "1%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "shortOutstandingPercent": {
        "label": "Short % Outstanding",
        "step": ["50%", "30%", "20%", "10%", "5%", "1%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "failToDeliver": {
        "label": "Fail to Deliver (FTD)",
        "step": ["1M", "500K", "200K", "100K", "50K", "10K", "1K"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "relativeFTD": {
        "label": "FTD / Avg. Volume",
        "step": ["300%", "200%", "100%", "50%", "20%", "10%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "freeCashFlow": {
        "label": "Free Cash Flow",
        "step": ["50B", "10B", "1B", "100M", "10M", "1M", 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "operatingCashFlow": {
        "label": "Operating Cash Flow",
        "step": ["50B", "10B", "1B", "100M", "10M", "1M", 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "operatingCashFlowPerShare": {
        "label": "Operating Cash Flow / Share",
        "step": [50, 40, 30, 10, 5, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "revenuePerShare": {
        "label": "Revenue / Share",
        "step": [50, 40, 30, 10, 5, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "netIncomePerShare": {
        "label": "Net Income / Share",
        "step": [50, 40, 30, 10, 5, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "shareholdersEquityPerShare": {
        "label": "Shareholders Equity / Share",
        "step": [50, 40, 30, 10, 5, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "interestDebtPerShare": {
        "label": "Interest Debt / Share",
        "step": [50, 40, 30, 10, 5, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "capexPerShare": {
        "label": "CapEx / Share",
        "step": [50, 40, 30, 10, 5, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "freeCashFlowMargin": {
        "label": "FCF Margin",
        "step": ["80%", "50%", "20%", "10%", "5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "totalDebt": {
        "label": "Total Debt",
        "step": ["200B", "100B", "50B", "10B", "1B", "100M", "10M", "1M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "operatingCashFlowSalesRatio": {
        "label": "Operating Cash Flow / Sales",
        "step": [5, 3, 1, 0.5, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "priceToOperatingCashFlowRatio": {
        "label": "Price / Cash Flow",
        "step": [20, 15, 10, 5, 3, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "priceToEarningsRatio": {
        "label": "PE Ratio",
        "step": [100, 50, 20, 10, 5, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "stockBasedCompensation": {
        "label": "Stock-Based Compensation",
        "step": ["10B", "1B", "100M", "10M", "1M", 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "stockBasedCompensationToRevenue": {
        "label": "SBC / Revenue",
        "step": ["20%", "10%", "5%", "1%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "totalStockholdersEquity": {
        "label": "Shareholders Equity",
        "step": ["100B", "50B", "10B", "1B", "100M", "50M", "10M", "1M", 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "sharesQoQ": {
        "label": "Shares Change (QoQ)",
        "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "sharesYoY": {
        "label": "Shares Change (YoY)",
        "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "grossProfitMargin": {
        "label": "Gross Margin",
        "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "netProfitMargin": {
        "label": "Profit Margin",
        "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "pretaxProfitMargin": {
        "label": "Pretax Margin",
        "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "ebitdaMargin": {
        "label": "EBITDA Margin",
        "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "ebitMargin": {
        "label": "EBIT Margin",
        "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "operatingMargin": {
        "label": "Operating Margin",
        "step": ["80%", "60%", "50%", "20%", "10%", "5%", "1%", "0.5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "interestIncomeToCapitalization": {
        "label": "Interest Income / Market Cap",
        "step": ["80%", "60%", "50%", "30%", "20%", "10%", "5%", "1%", "0.5%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "assetTurnover": {
        "label": "Asset Turnover",
        "step": [5, 3, 2, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "earningsYield": {
        "label": "Earnings Yield",
        "step": ["20%", "15%", "10%", "5%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "freeCashFlowYield": {
        "label": "FCF Yield",
        "step": ["20%", "15%", "10%", "5%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "effectiveTaxRate": {
        "label": "Effective Tax Rate",
        "step": ["20%", "15%", "10%", "5%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "fixedAssetTurnover": {
        "label": "Fixed Asset Turnover",
        "step": [10, 5, 3, 2, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "sharesOutStanding": {
        "label": "Shares Outstanding",
        "step": ["10B", "5B", "1B", "100M", "50M", "10M", "1M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "employees": {
        "label": "Employees",
        "step": ["500K", "300K", "200K", "100K", "10K", "1K", "100"],
        "defaultCondition": "over",
        "defaultValue": "100K",
    },
    "revenuePerEmployee": {
        "label": "Revenue Per Employee",
        "step": ["5M", "3M", "2M", "1M", "500K", "100K", 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "profitPerEmployee": {
        "label": "Profit Per Employee",
        "step": ["5M", "3M", "2M", "1M", "500K", "100K", 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "totalLiabilities": {
        "label": "Total Liabilities",
        "step": ["500B", "200B", "100B", "50B", "10B", "1B", "100M", "10M", "1M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "altmanZScore": {
        "label": "Altman-Z-Score",
        "step": [10, 5, 3, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "piotroskiScore": {
        "label": "Piotroski F-Score",
        "step": [9, 8, 7, 6, 5, 4, 3, 2, 1],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "earningsTime": {
        "label": "Earnings Time",
        "step": ["Before Market Open", "After Market Close"],
        "defaultCondition": "",
        "defaultValue": "any",
    },
    "earningsDate": {
        "label": "Earnings Date",
        "step": ["Today", "Tomorrow", "Next 7D", "Next 30D", "This Month", "Next Month"],
        "defaultCondition": "",
        "defaultValue": "any",
    },
    "earningsRevenueEst": {
        "label": "Earnings Revenue Estimate",
        "step": ["100B", "50B", "10B", "1B", "100M", "10M"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "earningsEPSEst": {
        "label": "Earnings EPS Estimate",
        "step": ["10", "5", "3", "2", "1", "0"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "earningsRevenueGrowthEst": {
        "label": "Revenue Estimated Growth",
        "step": ["100%", "50%", "20%", "10%", "5%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "earningsEPSGrowthEst": {
        "label": "EPS Estimated Growth",
        "step": ["100%", "50%", "20%", "10%", "5%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "analystRating": {
        "label": "Analyst Rating",
        "step": ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
        "defaultCondition": "",
        "defaultValue": "any",
    },
    "analystCounter": {
        "label": "Analyst Count",
        "step": ["40", "30", "20", "10", "5", "0"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "priceTarget": {
        "label": "Price Target",
        "step": ["1000", "500", "100", "10", "5", "1"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "upside": {
        "label": "Price Target Upside",
        "step": ["100%", "50%", "20%", "10%", "5%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "topAnalystRating": {
        "label": "Top Analyst Rating",
        "step": ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
        "defaultCondition": "",
        "defaultValue": "any",
    },
    "topAnalystCounter": {
        "label": "Top Analyst Count",
        "step": ["10", "5", "3", "1"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "topAnalystUpside": {
        "label": "Top Analyst Price Target Upside",
        "step": ["100%", "50%", "20%", "10%", "5%", "0%"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "topAnalystPriceTarget": {
        "label": "Top Analyst Price Target",
        "step": ["1000", "500", "100", "10", "5", "1"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "score": {
        "label": "AI Score",
        "step": ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
        "defaultCondition": "",
        "defaultValue": "any",
    },
    "sector": {
        "label": "Sector",
        "step": [
            "Basic Materials",
            "Communication Services",
            "Consumer Cyclical",
            "Consumer Defensive",
            "Energy",
            "Financial Services",
            "Healthcare",
            "Industrials",
            "Real Estate",
            "Technology",
            "Utilities",
        ],
        "defaultCondition": "",
        "defaultValue": "any",
    },
    "industry": {
        "label": "Industry",
        "step": industryList,
        "defaultCondition": "",
        "defaultValue": "any",
    },
    "country": {
        "label": "Country",
        "step": listOfRelevantCountries,
        "defaultCondition": "",
        "defaultValue": "any",
    },
    "ivRank": {
        "label": "IV Rank",
        "step": [50, 30, 20, 10, 5, 1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "iv30d": {
        "label": "IV 30d",
        "step": [1, 0.5, 0.3, 0.1, 0],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "totalOI": {
        "label": "Total OI",
        "step": ["500K", "300K", "200K", "100K", "50K", "10K", "1K"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "changeOI": {
        "label": "Change OI",
        "step": ["5K", "3K", "1K", "500", "300", "100"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "callVolume": {
        "label": "Call Volume",
        "step": ["100K", "50K", "20K", "10K", "5K", "1K"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "putVolume": {
        "label": "Put Volume",
        "step": ["100K", "50K", "20K", "10K", "5K", "1K"],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
    "pcRatio": {
        "label": "P/C Ratio",
        "step": [10, 5, 3, 2, 1, 0.5],
        "defaultCondition": "over",
        "defaultValue": "any",
    },
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
    """Build complete context of ALL available rules for the LLM"""
    
    context = "Complete Stock Screening Rules (174 total):\n\n"
    
    # Group rules for better organization
    rule_groups = {
        "Price & Volume": ['price', 'volume', 'avgVolume', 'relativeVolume', 'beta'],
        "Market & Size": ['marketCap', 'enterpriseValue', 'sharesOutStanding', 'employees'],
        "Short Selling": ['shortFloatPercent', 'shortRatio', 'sharesShort', 'shortOutstandingPercent', 'failToDeliver', 'relativeFTD'],
        "Dividends": ['dividendYield', 'dividendGrowth', 'annualDividend', 'payoutRatio', 'payoutFrequency'],
        "Valuation Ratios": ['pe', 'forwardPE', 'priceToBookRatio', 'priceToSalesRatio', 'forwardPS', 'priceToEarningsGrowthRatio', 'evToSales', 'evToEBITDA', 'evToEBIT', 'evToFreeCashFlow'],
        "Growth Metrics": ['growthRevenue', 'growthNetIncome', 'growthEPS', 'growthGrossProfit', 'growthEBITDA', 'growthFreeCashFlow', 'growthOperatingCashFlow'],
        "Technical Analysis": ['rsi', 'stochRSI', 'mfi', 'cci', 'atr', 'sma20', 'sma50', 'sma100', 'sma200', 'ema20', 'ema50', 'ema100', 'ema200'],
        "Financial Performance": ['returnOnAssets', 'returnOnEquity', 'returnOnTangibleAssets', 'returnOnInvestedCapital', 'returnOnCapitalEmployed'],
        "Cash Flow": ['freeCashFlow', 'operatingCashFlow', 'freeCashFlowPerShare', 'operatingCashFlowPerShare', 'priceToFreeCashFlowRatio', 'priceToOperatingCashFlowRatio', 'freeCashFlowMargin', 'freeCashFlowYield'],
        "Fundamentals": ['revenue', 'netIncome', 'grossProfit', 'ebit', 'ebitda', 'operatingIncome', 'eps', 'interestIncome', 'interestExpense'],
        "Company Info": ['sector', 'industry', 'country', 'revenuePerEmployee', 'profitPerEmployee']
    }
    
    # Show grouped rules first
    for group_name, rule_names in rule_groups.items():
        context += f"**{group_name}:**\n"
        for rule_name in rule_names:
            if rule_name in ALL_RULES:
                rule_data = ALL_RULES[rule_name]
                label = rule_data.get('label', rule_name)
                step_examples = rule_data.get('step', [])
                
                # Show first 2 examples to keep context manageable
                if len(step_examples) > 2:
                    examples = step_examples[:2] + ["..."]
                else:
                    examples = step_examples
                
                context += f"- {rule_name} ({label}): {examples}\n"
        context += "\n"
    
    # Add remaining rules that weren't in groups
    grouped_rules = set()
    for rule_names in rule_groups.values():
        grouped_rules.update(rule_names)
    
    remaining_rules = set(ALL_RULES.keys()) - grouped_rules
    if remaining_rules:
        context += "**Other Rules:**\n"
        for rule_name in sorted(remaining_rules):
            rule_data = ALL_RULES[rule_name]
            label = rule_data.get('label', rule_name)
            step_examples = rule_data.get('step', [])
            
            if len(step_examples) > 2:
                examples = step_examples[:2] + ["..."]
            else:
                examples = step_examples
            
            context += f"- {rule_name} ({label}): {examples}\n"
        context += "\n"
    
    context += f"""
**Total Available Rules:** {len(ALL_RULES)} rules

**Common Query Patterns:**
- "most shorted stocks" → shortFloatPercent > 20%, shortRatio > 1
- "price below $10" → price < 10  
- "large cap tech" → marketCap > 10B, sector = Technology
- "high dividend yield" → dividendYield > 3%
- "penny stocks" → price < 5
- "high volume stocks" → volume > 10M or avgVolume > 1M
- "growth stocks" → growthRevenue > 20%, growthEPS > 15%
- "value stocks" → pe < 15, priceToBookRatio < 2

**Conditions:** "over", "under", "exactly", "between"
**Values:** Use numbers for amounts (e.g., 10 for $10), strings with units (e.g., "10B" for billions), percentages as numbers (e.g., 20 for 20%)
"""
    
    return context

async def extract_screener_rules(query: str) -> Dict[str, Any]:
    try:
        rules_context = build_rules_context()
        
        system_prompt = f"""You are a stock screener expert. Extract screening rules from user queries.

{rules_context}

Extract only the specific rules that match the user's request. For "most shorted stocks", use shortFloatPercent and shortRatio rules. Only include rules explicitly mentioned or clearly implied by the user."""

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