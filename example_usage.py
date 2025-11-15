"""
Example usage of the FinancialDataFetcher class
"""

from data_fetcher import FinancialDataFetcher, quick_fetch

# Your API key
API_KEY = "8107eb9f9bmsh48137fb974399f6p1f3021jsn3ea14a6d79e2"

def main():
    # Initialize the fetcher
    fetcher = FinancialDataFetcher(API_KEY)
    
    # Example 1: Basic data fetching
    print("Fetching MSFT data...")
    msft_data = fetcher.fetch_intraday_data('MSFT', interval='5min')
    print(f"Fetched {len(msft_data)} data points for MSFT")
    print(msft_data.head())
    
    # Example 2: Add technical indicators
    print("\nAdding technical indicators...")
    msft_with_indicators = fetcher.add_technical_indicators(msft_data)
    print(f"Columns after adding indicators: {list(msft_with_indicators.columns)}")
    
    # Example 3: Time truncation
    print("\nTruncating to last 3 days...")
    recent_data = fetcher.truncate_by_time(msft_with_indicators, last_n_days=3)
    print(f"Data points after truncation: {len(recent_data)}")
    
    # Example 4: Aggregation to hourly data
    print("\nAggregating to hourly data...")
    hourly_data = fetcher.compute_aggregates(recent_data, timeframe='1H')
    print(f"Hourly data points: {len(hourly_data)}")
    print(hourly_data.head())
    
    # Example 5: Save data
    print("\nSaving data...")
    fetcher.save_data(hourly_data, 'data/msft_hourly.csv')
    
    # Example 6: Quick fetch (convenience function)
    print("\nUsing quick fetch for EURUSD...")
    eurusd_data = quick_fetch('EURUSD', API_KEY, days=5)
    print(f"EURUSD data shape: {eurusd_data.shape}")
    
    # Example 7: Multiple symbols
    print("\nFetching multiple symbols...")
    symbols = ['MSFT', 'AAPL']
    multi_data = fetcher.fetch_multiple_symbols(symbols, interval='5min')
    for symbol, data in multi_data.items():
        if data is not None:
            print(f"{symbol}: {len(data)} data points")

if __name__ == "__main__":
    main()