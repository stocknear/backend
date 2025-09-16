"""
DCF (Discounted Cash Flow) Model
Based on FastGraphs-style fundamental valuation approach
"""

import os
import numpy as np
import pandas as pd
import sqlite3
import asyncio
import aiohttp
import orjson
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dotenv import load_dotenv
from pathlib import Path
import logging
import aiofiles

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class DCFModel:
    """
    Discounted Cash Flow Model for stock valuation
    Implements a comprehensive DCF analysis similar to FastGraphs
    Uses local JSON financial statements for data
    """
    
    def __init__(self, api_key: Optional[str] = None, use_local_data: bool = True):
        """
        Initialize DCF Model
        
        Args:
            api_key: FMP API key for fetching financial data (optional if using local data)
            use_local_data: Whether to use local JSON files instead of API
        """
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.use_local_data = use_local_data
        self.json_base_path = Path("/home/mrahimi/stocknear/backend/app/json")
        self.risk_free_rate = 0.045  # Default 10-year Treasury rate (update periodically)
        self.market_risk_premium = 0.08  # Historical market risk premium
        
    async def fetch_financial_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch comprehensive financial data for DCF calculation
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing financial statements and metrics
        """
        data = {}
        
        if self.use_local_data:
            # Use local JSON files
            tasks = [
                self._fetch_local_income_statement(ticker),
                self._fetch_local_balance_sheet(ticker),
                self._fetch_local_cash_flow(ticker),
                self._fetch_local_profile(ticker),
                self._fetch_local_quote(ticker),
                self._fetch_local_key_metrics(ticker),
                self._fetch_local_enterprise_value(ticker)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            data['income_statement'] = results[0] if not isinstance(results[0], Exception) else []
            data['balance_sheet'] = results[1] if not isinstance(results[1], Exception) else []
            data['cash_flow'] = results[2] if not isinstance(results[2], Exception) else []
            data['profile'] = results[3] if not isinstance(results[3], Exception) else {}
            data['quote'] = results[4] if not isinstance(results[4], Exception) else {}
            data['key_metrics'] = results[5] if not isinstance(results[5], Exception) else []
            data['enterprise_value'] = results[6] if not isinstance(results[6], Exception) else []
        else:
            # Use API
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._fetch_income_statement(session, ticker),
                    self._fetch_balance_sheet(session, ticker),
                    self._fetch_cash_flow(session, ticker),
                    self._fetch_profile(session, ticker),
                    self._fetch_quote(session, ticker),
                    self._fetch_key_metrics(session, ticker),
                    self._fetch_enterprise_value(session, ticker)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                data['income_statement'] = results[0] if not isinstance(results[0], Exception) else []
                data['balance_sheet'] = results[1] if not isinstance(results[1], Exception) else []
                data['cash_flow'] = results[2] if not isinstance(results[2], Exception) else []
                data['profile'] = results[3] if not isinstance(results[3], Exception) else {}
                data['quote'] = results[4] if not isinstance(results[4], Exception) else {}
                data['key_metrics'] = results[5] if not isinstance(results[5], Exception) else []
                data['enterprise_value'] = results[6] if not isinstance(results[6], Exception) else []
            
        return data
    
    async def _fetch_income_statement(self, session: aiohttp.ClientSession, ticker: str) -> List[Dict]:
        """Fetch income statement data"""
        url = f"{self.base_url}/income-statement/{ticker}?limit=10&apikey={self.api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            return []
    
    async def _fetch_balance_sheet(self, session: aiohttp.ClientSession, ticker: str) -> List[Dict]:
        """Fetch balance sheet data"""
        url = f"{self.base_url}/balance-sheet-statement/{ticker}?limit=10&apikey={self.api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            return []
    
    async def _fetch_cash_flow(self, session: aiohttp.ClientSession, ticker: str) -> List[Dict]:
        """Fetch cash flow statement data"""
        url = f"{self.base_url}/cash-flow-statement/{ticker}?limit=10&apikey={self.api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            return []
    
    async def _fetch_profile(self, session: aiohttp.ClientSession, ticker: str) -> Dict:
        """Fetch company profile data"""
        url = f"{self.base_url}/profile/{ticker}?apikey={self.api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data[0] if data else {}
            return {}
    
    async def _fetch_quote(self, session: aiohttp.ClientSession, ticker: str) -> Dict:
        """Fetch current quote data"""
        url = f"{self.base_url}/quote/{ticker}?apikey={self.api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data[0] if data else {}
            return {}
    
    async def _fetch_key_metrics(self, session: aiohttp.ClientSession, ticker: str) -> List[Dict]:
        """Fetch key financial metrics"""
        url = f"{self.base_url}/key-metrics/{ticker}?limit=10&apikey={self.api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            return []
    
    async def _fetch_enterprise_value(self, session: aiohttp.ClientSession, ticker: str) -> List[Dict]:
        """Fetch enterprise value data"""
        url = f"{self.base_url}/enterprise-values/{ticker}?limit=10&apikey={self.api_key}"
        async with session.get(url) as response:
            if response.status == 200:
                return await response.json()
            return []
    
    # Local data fetching methods
    async def _fetch_local_income_statement(self, ticker: str) -> List[Dict]:
        """Fetch income statement from local JSON"""
        file_path = self.json_base_path / "financial-statements" / "income-statement" / "annual" / f"{ticker}.json"
        if file_path.exists():
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        return []
    
    async def _fetch_local_balance_sheet(self, ticker: str) -> List[Dict]:
        """Fetch balance sheet from local JSON"""
        file_path = self.json_base_path / "financial-statements" / "balance-sheet-statement" / "annual" / f"{ticker}.json"
        if file_path.exists():
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        return []
    
    async def _fetch_local_cash_flow(self, ticker: str) -> List[Dict]:
        """Fetch cash flow statement from local JSON"""
        file_path = self.json_base_path / "financial-statements" / "cash-flow-statement" / "annual" / f"{ticker}.json"
        if file_path.exists():
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        return []
    
    async def _fetch_local_profile(self, ticker: str) -> Dict:
        """Fetch company profile from local JSON or database"""
        # Try to get from profile JSON if exists
        file_path = self.json_base_path / "profile" / f"{ticker}.json"
        if file_path.exists():
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                return data[0] if isinstance(data, list) and data else data
        
        # Fallback: Try to get from stocks database
        db_path = Path("/home/mrahimi/stocknear/backend/app/stocks.db")
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT beta, marketCap, volAvg, shareOutstanding 
                FROM stocks 
                WHERE symbol = ?
            """, (ticker,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'beta': result[0],
                    'mktCap': result[1],
                    'volAvg': result[2],
                    'shareOutstanding': result[3]
                }
        return {}
    
    async def _fetch_local_quote(self, ticker: str) -> Dict:
        """Fetch quote data from local JSON or database"""
        # Try quote JSON
        file_path = self.json_base_path / "quote" / f"{ticker}.json"
        if file_path.exists():
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                return data[0] if isinstance(data, list) and data else data
    
        return {}
    
    async def _fetch_local_key_metrics(self, ticker: str) -> List[Dict]:
        """Fetch key metrics from local JSON"""
        file_path = self.json_base_path / "key-metrics" / f"{ticker}.json"
        if file_path.exists():
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        
        # Try ratios as alternative
        file_path = self.json_base_path / "financial-statements" / "ratios" / "annual" / f"{ticker}.json"
        if file_path.exists():
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        return []
    
    async def _fetch_local_enterprise_value(self, ticker: str) -> List[Dict]:
        """Fetch enterprise value from local JSON"""
        file_path = self.json_base_path / "enterprise-values" / f"{ticker}.json"
        if file_path.exists():
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                return json.loads(content)
        
        # Calculate enterprise value from available data if not found
        quote = await self._fetch_local_quote(ticker)
        balance_sheet = await self._fetch_local_balance_sheet(ticker)
        
        if quote and balance_sheet:
            market_cap = quote.get('marketCap', 0)
            if balance_sheet and len(balance_sheet) > 0:
                latest_bs = balance_sheet[0]
                total_debt = latest_bs.get('totalDebt', 0)
                cash = latest_bs.get('cashAndCashEquivalents', 0)
                
                enterprise_value = market_cap + total_debt - cash
                return [{
                    'enterpriseValue': enterprise_value,
                    'marketCapitalization': market_cap
                }]
        return []
    
    def calculate_free_cash_flow(self, financial_data: Dict[str, Any]) -> np.ndarray:
        """
        Calculate Free Cash Flow (FCF) from financial statements
        
        Args:
            financial_data: Dict containing financial statements
            
        Returns:
            Array of historical FCF values
        """
        cash_flow = financial_data.get('cash_flow', [])
        
        if not cash_flow:
            return np.array([])
        
        fcf_values = []
        
        for statement in cash_flow[:5]:  # Last 5 years
            # FCF = Operating Cash Flow - Capital Expenditures
            operating_cf = statement.get('operatingCashFlow', 0)
            capex = abs(statement.get('capitalExpenditure', 0))
            
            fcf = operating_cf - capex
            fcf_values.append(fcf)
        
        return np.array(fcf_values[::-1])  # Reverse to get chronological order
    
    def calculate_owner_earnings(self, financial_data: Dict[str, Any]) -> np.ndarray:
        """
        Calculate Owner Earnings (Buffett's preferred metric)
        Owner Earnings = Net Income + Depreciation & Amortization - CapEx - Change in Working Capital
        
        Args:
            financial_data: Dict containing financial statements
            
        Returns:
            Array of owner earnings values
        """
        income_statements = financial_data.get('income_statement', [])
        cash_flow_statements = financial_data.get('cash_flow', [])
        
        if not income_statements or not cash_flow_statements:
            return np.array([])
        
        owner_earnings = []
        
        for i in range(min(5, len(income_statements), len(cash_flow_statements))):
            net_income = income_statements[i].get('netIncome', 0)
            depreciation = cash_flow_statements[i].get('depreciationAndAmortization', 0)
            capex = abs(cash_flow_statements[i].get('capitalExpenditure', 0))
            working_capital_change = cash_flow_statements[i].get('changeInWorkingCapital', 0)
            
            owner_earning = net_income + depreciation - capex - working_capital_change
            owner_earnings.append(owner_earning)
        
        return np.array(owner_earnings[::-1])
    
    def calculate_growth_rates(self, cash_flows: np.ndarray) -> Dict[str, float]:
        """
        Calculate various growth rate scenarios
        
        Args:
            cash_flows: Historical cash flow values
            
        Returns:
            Dict with different growth rate calculations
        """
        if len(cash_flows) < 2:
            return {
                'historical_average': 0.03,
                'cagr': 0.03,
                'linear_regression': 0.03,
                'conservative': 0.02,
                'moderate': 0.05,
                'optimistic': 0.08
            }
        
        # Historical average growth rate
        growth_rates = []
        for i in range(1, len(cash_flows)):
            if cash_flows[i-1] > 0:
                growth_rate = (cash_flows[i] - cash_flows[i-1]) / cash_flows[i-1]
                growth_rates.append(growth_rate)
        
        historical_avg = np.mean(growth_rates) if growth_rates else 0.03
        
        # CAGR (Compound Annual Growth Rate)
        if cash_flows[0] > 0 and cash_flows[-1] > 0:
            years = len(cash_flows) - 1
            cagr = (cash_flows[-1] / cash_flows[0]) ** (1/years) - 1
        else:
            cagr = 0.03
        
        # Linear regression growth rate
        x = np.arange(len(cash_flows))
        if len(cash_flows) >= 3:
            coefficients = np.polyfit(x, cash_flows, 1)
            linear_growth = coefficients[0] / np.mean(cash_flows) if np.mean(cash_flows) > 0 else 0.03
        else:
            linear_growth = historical_avg
        
        # Apply bounds to growth rates
        historical_avg = np.clip(historical_avg, -0.5, 0.5)
        cagr = np.clip(cagr, -0.5, 0.5)
        linear_growth = np.clip(linear_growth, -0.5, 0.5)
        
        return {
            'historical_average': historical_avg,
            'cagr': cagr,
            'linear_regression': linear_growth,
            'conservative': min(historical_avg, cagr) * 0.7,  # 70% of lower growth rate
            'moderate': np.mean([historical_avg, cagr]),
            'optimistic': max(historical_avg, cagr) * 1.2  # 120% of higher growth rate
        }
    
    def calculate_wacc(self, financial_data: Dict[str, Any], beta: Optional[float] = None) -> float:
        """
        Calculate Weighted Average Cost of Capital (WACC)
        
        Args:
            financial_data: Dict containing financial data
            beta: Beta coefficient (if not provided, will be extracted from data)
            
        Returns:
            WACC as a decimal
        """
        profile = financial_data.get('profile', {})
        balance_sheet = financial_data.get('balance_sheet', [])
        key_metrics = financial_data.get('key_metrics', [])
        
        # Get beta
        if beta is None:
            beta = profile.get('beta', 1.0)
            if beta is None or beta <= 0:
                beta = 1.0  # Default to market beta
        
        # Calculate cost of equity using CAPM
        cost_of_equity = self.risk_free_rate + beta * self.market_risk_premium
        
        # Get debt and equity values
        if balance_sheet:
            latest_bs = balance_sheet[0]
            total_debt = latest_bs.get('totalDebt', 0)
            
            # Get market cap
            quote = financial_data.get('quote', {})
            market_cap = quote.get('marketCap', 0)
            
            if market_cap == 0 and key_metrics:
                market_cap = key_metrics[0].get('marketCap', 0)
            
            # If still no market cap, use book value of equity
            if market_cap == 0:
                market_cap = latest_bs.get('totalStockholdersEquity', 0)
            
            # Calculate weights
            total_value = market_cap + total_debt
            
            if total_value > 0:
                weight_equity = market_cap / total_value
                weight_debt = total_debt / total_value
            else:
                weight_equity = 1.0
                weight_debt = 0.0
            
            # Estimate cost of debt
            income_statement = financial_data.get('income_statement', [])
            if income_statement and total_debt > 0:
                interest_expense = abs(income_statement[0].get('interestExpense', 0))
                cost_of_debt = interest_expense / total_debt if total_debt > 0 else 0.03
            else:
                cost_of_debt = 0.03  # Default cost of debt
            
            # Get tax rate
            if income_statement:
                ebt = income_statement[0].get('incomeBeforeTax', 0)
                tax = income_statement[0].get('incomeTaxExpense', 0)
                tax_rate = tax / ebt if ebt > 0 else 0.21  # Default corporate tax rate
            else:
                tax_rate = 0.21
            
            # Calculate WACC
            wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))
            
        else:
            # If no balance sheet data, use cost of equity as WACC
            wacc = cost_of_equity
        
        # Apply reasonable bounds to WACC
        wacc = np.clip(wacc, 0.05, 0.20)
        
        return wacc
    
    def calculate_terminal_value(self, 
                                final_fcf: float, 
                                terminal_growth_rate: float, 
                                wacc: float) -> float:
        """
        Calculate terminal value using perpetuity growth method
        
        Args:
            final_fcf: Final year free cash flow
            terminal_growth_rate: Perpetual growth rate
            wacc: Weighted average cost of capital
            
        Returns:
            Terminal value
        """
        if wacc <= terminal_growth_rate:
            # If WACC is too close to growth rate, adjust
            wacc = terminal_growth_rate + 0.02
        
        terminal_value = final_fcf * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
        
        return max(0, terminal_value)  # Terminal value cannot be negative
    
    def project_cash_flows(self, 
                          base_fcf: float, 
                          growth_rates: List[float], 
                          projection_years: int = 10) -> np.ndarray:
        """
        Project future cash flows based on growth rates
        
        Args:
            base_fcf: Starting free cash flow
            growth_rates: List of growth rates for each year (can be single rate or multiple)
            projection_years: Number of years to project
            
        Returns:
            Array of projected cash flows
        """
        projected_fcf = [base_fcf]
        
        # Handle single growth rate or multiple rates
        if isinstance(growth_rates, (int, float)):
            growth_rates = [growth_rates] * projection_years
        elif len(growth_rates) < projection_years:
            # Extend with the last growth rate
            growth_rates.extend([growth_rates[-1]] * (projection_years - len(growth_rates)))
        
        for i in range(projection_years):
            next_fcf = projected_fcf[-1] * (1 + growth_rates[i])
            projected_fcf.append(next_fcf)
        
        return np.array(projected_fcf[1:])  # Exclude base year
    
    def calculate_dcf_value(self,
                           projected_fcf: np.ndarray,
                           terminal_value: float,
                           wacc: float,
                           shares_outstanding: float) -> Dict[str, float]:
        """
        Calculate the DCF valuation
        
        Args:
            projected_fcf: Array of projected free cash flows
            terminal_value: Terminal value at end of projection period
            wacc: Discount rate (WACC)
            shares_outstanding: Number of shares outstanding
            
        Returns:
            Dict with enterprise value, equity value, and per-share value
        """
        # Calculate present value of projected cash flows
        pv_fcf = 0
        for i, fcf in enumerate(projected_fcf):
            pv_fcf += fcf / ((1 + wacc) ** (i + 1))
        
        # Calculate present value of terminal value
        pv_terminal = terminal_value / ((1 + wacc) ** len(projected_fcf))
        
        # Enterprise value
        enterprise_value = pv_fcf + pv_terminal
        
        # Per share value (simplified - would need to adjust for net debt for equity value)
        if shares_outstanding > 0:
            value_per_share = enterprise_value / shares_outstanding
        else:
            value_per_share = 0
        
        return {
            'enterprise_value': enterprise_value,
            'pv_cash_flows': pv_fcf,
            'pv_terminal_value': pv_terminal,
            'value_per_share': value_per_share
        }
    
    def two_stage_dcf(self,
                     base_fcf: float,
                     high_growth_rate: float,
                     high_growth_years: int,
                     stable_growth_rate: float,
                     wacc: float,
                     shares_outstanding: float) -> Dict[str, float]:
        """
        Two-stage DCF model with high growth followed by stable growth
        
        Args:
            base_fcf: Current free cash flow
            high_growth_rate: Growth rate for initial years
            high_growth_years: Number of high growth years
            stable_growth_rate: Terminal growth rate
            wacc: Discount rate
            shares_outstanding: Number of shares
            
        Returns:
            DCF valuation results
        """
        # Stage 1: High growth period
        stage1_fcf = []
        current_fcf = base_fcf
        
        for i in range(high_growth_years):
            current_fcf *= (1 + high_growth_rate)
            stage1_fcf.append(current_fcf)
        
        # Stage 2: Stable growth (terminal value)
        terminal_fcf = stage1_fcf[-1] if stage1_fcf else base_fcf
        terminal_value = self.calculate_terminal_value(terminal_fcf, stable_growth_rate, wacc)
        
        # Calculate DCF value
        return self.calculate_dcf_value(
            np.array(stage1_fcf),
            terminal_value,
            wacc,
            shares_outstanding
        )
    
    def sensitivity_analysis(self,
                           base_fcf: float,
                           base_growth_rate: float,
                           base_wacc: float,
                           terminal_growth: float,
                           shares_outstanding: float,
                           wacc_range: Tuple[float, float] = (-0.02, 0.02),
                           growth_range: Tuple[float, float] = (-0.02, 0.02),
                           steps: int = 5) -> pd.DataFrame:
        """
        Perform sensitivity analysis on DCF valuation
        
        Args:
            base_fcf: Base free cash flow
            base_growth_rate: Base growth rate assumption
            base_wacc: Base WACC assumption  
            terminal_growth: Terminal growth rate
            shares_outstanding: Number of shares
            wacc_range: Range to vary WACC (as deviation from base)
            growth_range: Range to vary growth rate (as deviation from base)
            steps: Number of steps in each direction
            
        Returns:
            DataFrame with sensitivity analysis results
        """
        wacc_values = np.linspace(
            base_wacc + wacc_range[0],
            base_wacc + wacc_range[1],
            steps
        )
        
        growth_values = np.linspace(
            base_growth_rate + growth_range[0],
            base_growth_rate + growth_range[1],
            steps
        )
        
        results = np.zeros((len(wacc_values), len(growth_values)))
        
        for i, wacc in enumerate(wacc_values):
            for j, growth in enumerate(growth_values):
                # Project cash flows
                projected_fcf = self.project_cash_flows(base_fcf, growth, 10)
                
                # Calculate terminal value
                terminal_value = self.calculate_terminal_value(
                    projected_fcf[-1],
                    terminal_growth,
                    wacc
                )
                
                # Calculate DCF value
                dcf_result = self.calculate_dcf_value(
                    projected_fcf,
                    terminal_value,
                    wacc,
                    shares_outstanding
                )
                
                results[i, j] = dcf_result['value_per_share']
        
        # Create DataFrame
        sensitivity_df = pd.DataFrame(
            results,
            index=[f"WACC: {wacc:.1%}" for wacc in wacc_values],
            columns=[f"Growth: {growth:.1%}" for growth in growth_values]
        )
        
        return sensitivity_df
    
    def monte_carlo_simulation(self,
                             base_fcf: float,
                             growth_mean: float,
                             growth_std: float,
                             wacc_mean: float,
                             wacc_std: float,
                             terminal_growth_mean: float,
                             terminal_growth_std: float,
                             shares_outstanding: float,
                             simulations: int = 10000) -> Dict[str, Any]:
        """
        Monte Carlo simulation for DCF valuation
        
        Args:
            base_fcf: Base free cash flow
            growth_mean: Mean growth rate
            growth_std: Standard deviation of growth rate
            wacc_mean: Mean WACC
            wacc_std: Standard deviation of WACC
            terminal_growth_mean: Mean terminal growth rate
            terminal_growth_std: Standard deviation of terminal growth
            shares_outstanding: Number of shares
            simulations: Number of Monte Carlo simulations
            
        Returns:
            Dict with simulation results and statistics
        """
        np.random.seed(42)  # For reproducibility
        
        valuations = []
        
        for _ in range(simulations):
            # Sample random parameters
            growth_rate = np.random.normal(growth_mean, growth_std)
            wacc = np.random.normal(wacc_mean, wacc_std)
            terminal_growth = np.random.normal(terminal_growth_mean, terminal_growth_std)
            
            # Ensure reasonable bounds
            growth_rate = np.clip(growth_rate, -0.5, 0.5)
            wacc = np.clip(wacc, 0.01, 0.30)
            terminal_growth = np.clip(terminal_growth, 0, 0.05)
            
            # Skip if WACC <= terminal growth
            if wacc <= terminal_growth:
                continue
            
            # Project cash flows
            projected_fcf = self.project_cash_flows(base_fcf, growth_rate, 10)
            
            # Calculate terminal value
            terminal_value = self.calculate_terminal_value(
                projected_fcf[-1],
                terminal_growth,
                wacc
            )
            
            # Calculate DCF value
            dcf_result = self.calculate_dcf_value(
                projected_fcf,
                terminal_value,
                wacc,
                shares_outstanding
            )
            
            valuations.append(dcf_result['value_per_share'])
        
        valuations = np.array(valuations)
        
        return {
            'mean_value': np.mean(valuations),
            'median_value': np.median(valuations),
            'std_dev': np.std(valuations),
            'min_value': np.min(valuations),
            'max_value': np.max(valuations),
            'percentile_5': np.percentile(valuations, 5),
            'percentile_25': np.percentile(valuations, 25),
            'percentile_75': np.percentile(valuations, 75),
            'percentile_95': np.percentile(valuations, 95),
            'valuations': valuations
        }
    
    async def full_dcf_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Perform complete DCF analysis for a stock
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Comprehensive DCF analysis results
        """
        logger.info(f"Starting DCF analysis for {ticker}")
        
        # Fetch financial data
        financial_data = await self.fetch_financial_data(ticker)
        
        # Extract key information
        profile = financial_data.get('profile', {})
        quote = financial_data.get('quote', {})
        
        shares_outstanding = quote.get('sharesOutstanding', 0)
        if shares_outstanding == 0:
            shares_outstanding = profile.get('shareOutstanding', 0)
        
        current_price = quote.get('price', 0)
        beta = profile.get('beta', 1.0)
        
        # Calculate free cash flow
        fcf_values = self.calculate_free_cash_flow(financial_data)
        owner_earnings = self.calculate_owner_earnings(financial_data)
        
        # Use the most recent positive FCF or owner earnings
        if len(fcf_values) > 0 and fcf_values[-1] > 0:
            base_fcf = fcf_values[-1]
            historical_fcf = fcf_values
        elif len(owner_earnings) > 0 and owner_earnings[-1] > 0:
            base_fcf = owner_earnings[-1]
            historical_fcf = owner_earnings
        else:
            logger.warning(f"No positive FCF found for {ticker}")
            return {
                'error': 'No positive free cash flow found',
                'ticker': ticker
            }
        
        # Calculate growth rates
        growth_rates = self.calculate_growth_rates(historical_fcf)
        
        # Calculate WACC
        wacc = self.calculate_wacc(financial_data, beta)
        
        # Terminal growth rate (conservative: GDP growth rate)
        terminal_growth = 0.025  # 2.5% perpetual growth
        
        # Perform valuations with different scenarios
        scenarios = {}
        
        # Conservative scenario
        conservative_growth = growth_rates['conservative']
        conservative_fcf = self.project_cash_flows(base_fcf, conservative_growth, 10)
        conservative_terminal = self.calculate_terminal_value(
            conservative_fcf[-1], terminal_growth, wacc
        )
        scenarios['conservative'] = self.calculate_dcf_value(
            conservative_fcf, conservative_terminal, wacc, shares_outstanding
        )
        
        # Moderate scenario
        moderate_growth = growth_rates['moderate']
        moderate_fcf = self.project_cash_flows(base_fcf, moderate_growth, 10)
        moderate_terminal = self.calculate_terminal_value(
            moderate_fcf[-1], terminal_growth, wacc
        )
        scenarios['moderate'] = self.calculate_dcf_value(
            moderate_fcf, moderate_terminal, wacc, shares_outstanding
        )
        
        # Optimistic scenario
        optimistic_growth = growth_rates['optimistic']
        optimistic_fcf = self.project_cash_flows(base_fcf, optimistic_growth, 10)
        optimistic_terminal = self.calculate_terminal_value(
            optimistic_fcf[-1], terminal_growth, wacc
        )
        scenarios['optimistic'] = self.calculate_dcf_value(
            optimistic_fcf, optimistic_terminal, wacc, shares_outstanding
        )
        
        # Two-stage model
        two_stage = self.two_stage_dcf(
            base_fcf,
            growth_rates['cagr'],  # Use historical CAGR for high growth
            5,  # 5 years of high growth
            terminal_growth,
            wacc,
            shares_outstanding
        )
        
        # Sensitivity analysis
        sensitivity = self.sensitivity_analysis(
            base_fcf,
            moderate_growth,
            wacc,
            terminal_growth,
            shares_outstanding
        )
        
        # Monte Carlo simulation
        monte_carlo = self.monte_carlo_simulation(
            base_fcf,
            moderate_growth,
            abs(moderate_growth) * 0.3,  # 30% of growth rate as std dev
            wacc,
            wacc * 0.2,  # 20% of WACC as std dev
            terminal_growth,
            0.005,  # 0.5% std dev for terminal growth
            shares_outstanding,
            simulations=5000
        )
        
        # Calculate implied metrics
        results = {
            'ticker': ticker,
            'current_price': current_price,
            'shares_outstanding': shares_outstanding,
            'base_fcf': base_fcf,
            'historical_fcf': historical_fcf.tolist() if len(historical_fcf) > 0 else [],
            'wacc': wacc,
            'beta': beta,
            'growth_rates': growth_rates,
            'terminal_growth_rate': terminal_growth,
            'scenarios': scenarios,
            'two_stage_model': two_stage,
            'sensitivity_analysis': sensitivity.to_dict(),
            'monte_carlo': {
                'mean_value': monte_carlo['mean_value'],
                'median_value': monte_carlo['median_value'],
                'std_dev': monte_carlo['std_dev'],
                'percentile_5': monte_carlo['percentile_5'],
                'percentile_95': monte_carlo['percentile_95']
            },
            'valuation_summary': {
                'conservative_value': scenarios['conservative']['value_per_share'],
                'moderate_value': scenarios['moderate']['value_per_share'],
                'optimistic_value': scenarios['optimistic']['value_per_share'],
                'two_stage_value': two_stage['value_per_share'],
                'monte_carlo_mean': monte_carlo['mean_value'],
                'current_price': current_price,
                'average_fair_value': np.mean([
                    scenarios['conservative']['value_per_share'],
                    scenarios['moderate']['value_per_share'],
                    scenarios['optimistic']['value_per_share'],
                    two_stage['value_per_share']
                ])
            }
        }
        
        # Calculate upside/downside
        avg_fair_value = results['valuation_summary']['average_fair_value']
        if current_price > 0:
            results['valuation_summary']['upside_percentage'] = (
                (avg_fair_value - current_price) / current_price * 100
            )
            results['valuation_summary']['price_to_fair_value'] = current_price / avg_fair_value
        
        logger.info(f"DCF analysis completed for {ticker}")
        
        return results
    
    def calculate_reverse_dcf(self,
                            current_price: float,
                            base_fcf: float,
                            wacc: float,
                            terminal_growth: float,
                            shares_outstanding: float,
                            years: int = 10) -> float:
        """
        Reverse DCF: Calculate implied growth rate from current market price
        
        Args:
            current_price: Current stock price
            base_fcf: Current free cash flow
            wacc: Discount rate
            terminal_growth: Terminal growth rate
            shares_outstanding: Number of shares
            years: Projection period
            
        Returns:
            Implied growth rate
        """
        target_enterprise_value = current_price * shares_outstanding
        
        # Binary search for implied growth rate
        low, high = -0.5, 0.5
        tolerance = 0.0001
        max_iterations = 100
        
        for _ in range(max_iterations):
            mid = (low + high) / 2
            
            # Calculate DCF with this growth rate
            projected_fcf = self.project_cash_flows(base_fcf, mid, years)
            terminal_value = self.calculate_terminal_value(
                projected_fcf[-1], terminal_growth, wacc
            )
            dcf_result = self.calculate_dcf_value(
                projected_fcf, terminal_value, wacc, shares_outstanding
            )
            
            calculated_value = dcf_result['enterprise_value']
            
            if abs(calculated_value - target_enterprise_value) < tolerance:
                return mid
            elif calculated_value < target_enterprise_value:
                low = mid
            else:
                high = mid
        
        return mid


# Example usage and testing
async def main():
    """Example usage of the DCF Model"""
    dcf = DCFModel()
    
    # Example: Analyze Apple Inc.
    ticker = "AAPL"
    
    try:
        results = await dcf.full_dcf_analysis(ticker)
        
        print(f"\n{'='*60}")
        print(f"DCF Analysis for {ticker}")
        print(f"{'='*60}")
        
        print(f"\nCurrent Price: ${results['current_price']:.2f}")
        print(f"WACC: {results['wacc']:.2%}")
        print(f"Beta: {results['beta']:.2f}")
        
        print(f"\nValuation Scenarios:")
        print(f"  Conservative: ${results['valuation_summary']['conservative_value']:.2f}")
        print(f"  Moderate: ${results['valuation_summary']['moderate_value']:.2f}")
        print(f"  Optimistic: ${results['valuation_summary']['optimistic_value']:.2f}")
        print(f"  Two-Stage: ${results['valuation_summary']['two_stage_value']:.2f}")
        
        print(f"\nMonte Carlo Analysis:")
        print(f"  Mean Value: ${results['valuation_summary']['monte_carlo_mean']:.2f}")
        
        print(f"\nFair Value Summary:")
        print(f"  Average Fair Value: ${results['valuation_summary']['average_fair_value']:.2f}")
        
        if 'upside_percentage' in results['valuation_summary']:
            print(f"  Upside/Downside: {results['valuation_summary']['upside_percentage']:.1f}%")
            print(f"  Price to Fair Value: {results['valuation_summary']['price_to_fair_value']:.2f}")
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        logger.error(f"Error in DCF analysis: {str(e)}", exc_info=True)


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())