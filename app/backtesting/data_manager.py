"""
Data manager for efficient loading and caching of market data.
Handles data loading, caching, and column standardization.
"""

import pandas as pd
import aiofiles
import orjson
from pathlib import Path
from typing import Dict, Optional, List
from functools import lru_cache


class DataManager:
    """Manages market data loading with caching and optimization"""
    
    def __init__(self, market_data_path: str = "json/historical-price/adj"):
        self.market_data_path = market_data_path
        self._cache = {}
    
    async def load_historical_data(self, ticker: str, start_date: str = None, 
                                   end_date: str = None) -> pd.DataFrame:
        """
        Load historical data for a single ticker with caching
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with historical data
        """
        # Create cache key
        cache_key = f"{ticker}_{start_date}_{end_date}"
        
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Try different file locations
        possible_paths = [
            Path(f"json/historical-price/adj/{ticker}.json"), 
        ]
        
        for file_path in possible_paths:
            if file_path.exists():
                try:
                    df = await self._load_and_process_file(file_path)
                    if not df.empty:
                        # Filter by date range if specified
                        df = self._filter_by_date_range(df, start_date, end_date, ticker)
                        if not df.empty:
                            # Cache the result
                            self._cache[cache_key] = df.copy()
                            return df
                except Exception as e:
                    print(f"Error loading data from {file_path}: {str(e)}")
                    continue
        
        print(f"Error: No historical data file found for {ticker}")
        return pd.DataFrame()
    
    async def _load_and_process_file(self, file_path: Path) -> pd.DataFrame:
        """Load and process a single data file"""
        async with aiofiles.open(file_path, 'rb') as f:
            content = await f.read()
            data = orjson.loads(content)
        
        df = pd.DataFrame(data)
        if df.empty:
            return df
            
        # Convert date column to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Standardize column names for compatibility
        self._standardize_columns(df)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> None:
        """Standardize column names in place"""
        column_mapping = {
            'adjClose': 'close',
            'adjOpen': 'open', 
            'adjHigh': 'high',
            'adjLow': 'low'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
    
    def _filter_by_date_range(self, df: pd.DataFrame, start_date: str, 
                              end_date: str, ticker: str) -> pd.DataFrame:
        """Filter dataframe by date range"""
        if start_date:
            start_dt = pd.to_datetime(start_date)
            # Find closest available date
            closest_start = self._find_closest_date(df, start_dt, 'forward')
            if closest_start is not None:
                df = df[df.index >= closest_start]
                actual_start = closest_start.strftime('%Y-%m-%d')
                if actual_start != start_date:
                    print(f"Start date adjusted from {start_date} to {actual_start}")
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            # Find closest available date
            closest_end = self._find_closest_date(df, end_dt, 'backward')
            if closest_end is not None:
                df = df[df.index <= closest_end]
        
        if not df.empty:
            print(f"Successfully loaded {len(df)} data points for {ticker}")
            print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        return df
    
    def _find_closest_date(self, df: pd.DataFrame, target_date: pd.Timestamp, 
                           direction: str = 'forward') -> Optional[pd.Timestamp]:
        """Find the closest available date in the dataframe"""
        if df.empty or target_date in df.index:
            return target_date if not df.empty else None
            
        if direction == 'forward':
            future_dates = df.index[df.index > target_date]
            return future_dates[0] if len(future_dates) > 0 else None
        else:  # backward
            past_dates = df.index[df.index < target_date]
            return past_dates[-1] if len(past_dates) > 0 else None
    
    async def load_multiple_tickers(self, tickers: List[str], start_date: str = None, 
                                    end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Load historical data for multiple tickers efficiently"""
        data_dict = {}
        
        for ticker in tickers:
            try:
                df = await self.load_historical_data(ticker, start_date, end_date)
                if not df.empty:
                    data_dict[ticker] = df
            except Exception as e:
                print(f"Error loading data for {ticker}: {str(e)}")
        
        return data_dict
    
    def clear_cache(self):
        """Clear the data cache"""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached datasets"""
        return len(self._cache)