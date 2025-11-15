"""
Financial Data Fetcher for ML Model Training

A flexible class for fetching and processing financial data from Alpha Vantage API
with support for aggregations, time truncation, and extensible features.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Union
import json
import time


class FinancialDataFetcher:
    """
    Flexible financial data fetcher with support for various data sources and processing options.
    
    Features:
    - Fetch intraday data from Alpha Vantage API
    - Time-based data truncation
    - Data aggregation capabilities
    - Extensible for multiple symbols and timeframes
    - Built-in rate limiting and error handling
    """
    
    def __init__(self, api_key: str, base_url: str = "https://alpha-vantage.p.rapidapi.com/query"):
        """
        Initialize the data fetcher.
        
        Args:
            api_key: RapidAPI key for Alpha Vantage
            base_url: Base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'X-RapidAPI-Key': api_key,
            'X-RapidAPI-Host': 'alpha-vantage.p.rapidapi.com'
        }
        self.last_request_time = 0
        self.rate_limit_delay = 1  # seconds between requests
    
    def _make_request(self, params: Dict) -> Dict:
        """
        Make API request with rate limiting and error handling.
        
        Args:
            params: Request parameters
            
        Returns:
            JSON response as dictionary
        """
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        try:
            response = requests.get(self.base_url, headers=self.headers, params=params)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
    
    def fetch_intraday_data(self, 
                           symbol: str, 
                           interval: str = '5min',
                           output_size: str = 'compact') -> pd.DataFrame:
        """
        Fetch intraday time series data for a symbol.
        
        Args:
            symbol: Stock/forex symbol (e.g., 'MSFT', 'EURUSD')
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            output_size: 'compact' (last 100 data points) or 'full' (20+ years)
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "datatype": "json",
            "output_size": output_size
        }
        
        data = self._make_request(params)
        
        # Extract time series data
        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            raise ValueError(f"No time series data found. Response: {data}")
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Clean column names and convert to numeric
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Add metadata
        df.attrs['symbol'] = symbol
        df.attrs['interval'] = interval
        df.attrs['meta_data'] = data.get('Meta Data', {})
        
        return df 
   
    def truncate_by_time(self, 
                        df: pd.DataFrame, 
                        start_date: Optional[Union[str, datetime]] = None,
                        end_date: Optional[Union[str, datetime]] = None,
                        last_n_days: Optional[int] = None) -> pd.DataFrame:
        """
        Truncate data by time range or last N days.
        
        Args:
            df: Input DataFrame with datetime index
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)
            last_n_days: Keep only last N days of data
            
        Returns:
            Truncated DataFrame
        """
        if last_n_days:
            cutoff_date = df.index.max() - timedelta(days=last_n_days)
            return df[df.index >= cutoff_date]
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df.index >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df.index <= end_date]
        
        return df
    
    def compute_aggregates(self, 
                          df: pd.DataFrame, 
                          timeframe: str = '1H',
                          agg_functions: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Compute time-based aggregates (e.g., hourly, daily).
        
        Args:
            df: Input DataFrame with OHLCV data
            timeframe: Pandas frequency string ('1H', '1D', '1W', etc.)
            agg_functions: Custom aggregation functions per column
            
        Returns:
            Aggregated DataFrame
        """
        if agg_functions is None:
            agg_functions = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
        
        return df.resample(timeframe).agg(agg_functions).dropna()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical indicators to the dataset.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        df = df.copy()
        
        # Price-based indicators
        df['price_change'] = df['close'].pct_change()
        df['price_range'] = df['high'] - df['low']
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        
        # Volatility
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def fetch_multiple_symbols(self, 
                              symbols: List[str], 
                              interval: str = '5min',
                              **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of symbols to fetch
            interval: Time interval
            **kwargs: Additional arguments for fetch_intraday_data
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        for symbol in symbols:
            try:
                print(f"Fetching data for {symbol}...")
                results[symbol] = self.fetch_intraday_data(symbol, interval, **kwargs)
                time.sleep(self.rate_limit_delay)  # Rate limiting
            except Exception as e:
                print(f"Failed to fetch data for {symbol}: {e}")
                results[symbol] = None
        
        return results
    
    def save_data(self, df: pd.DataFrame, filepath: str, format: str = 'csv') -> None:
        """
        Save DataFrame to file.
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            format: File format ('csv', 'parquet', 'json')
        """
        if format.lower() == 'csv':
            df.to_csv(filepath)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='index', date_format='iso')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Data saved to {filepath}")
    
    def load_data(self, filepath: str, format: str = 'csv') -> pd.DataFrame:
        """
        Load DataFrame from file.
        
        Args:
            filepath: Input file path
            format: File format ('csv', 'parquet', 'json')
            
        Returns:
            Loaded DataFrame
        """
        if format.lower() == 'csv':
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif format.lower() == 'parquet':
            df = pd.read_parquet(filepath)
        elif format.lower() == 'json':
            df = pd.read_json(filepath, orient='index')
            df.index = pd.to_datetime(df.index)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return df


# Example usage and convenience functions
def create_forex_fetcher(api_key: str) -> FinancialDataFetcher:
    """Create a fetcher instance configured for forex data."""
    return FinancialDataFetcher(api_key)


def quick_fetch(symbol: str, api_key: str, days: int = 7) -> pd.DataFrame:
    """
    Quick fetch function for getting recent data with basic indicators.
    
    Args:
        symbol: Symbol to fetch
        api_key: API key
        days: Number of recent days to keep
        
    Returns:
        DataFrame with data and technical indicators
    """
    fetcher = FinancialDataFetcher(api_key)
    df = fetcher.fetch_intraday_data(symbol)
    df = fetcher.truncate_by_time(df, last_n_days=days)
    df = fetcher.add_technical_indicators(df)
    return df