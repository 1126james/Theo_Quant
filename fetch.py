"""
Cryptocurrency Historical Data Fetcher

This script fetches historical OHLCV (Open, High, Low, Close, Volume) data from 
Binance and Upbit exchanges using the CCXT library. It supports data caching and multiple 
timeframes, from 1-minute to monthly candles.

Features:
- Fetches BTC/USDT:USDT perpetual futures data from Binance
- Fetches BTC/KRW spot data from Upbit
- Supports multiple timeframes (1m to 1M)
- Implements data caching to avoid redundant API calls
- Handles rate limiting and exchange-specific restrictions
- Progress tracking during data fetching
- Command-line interface for easy usage

Usage:
    python fetch.py -s 2021-1-1 -e 2023-1-1 -t daily -d data
    (above is default value)
    or
    python fetch.py
"""

from typing import Optional, List, Dict, Union, Any
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import argparse
from tqdm import tqdm

def parse_date(date_str: str) -> datetime:
    """
    Converts string date to datetime object.
    
    Args:
        date_str: Date string in YYYY-M-D format
        
    Returns:
        Parsed datetime object
        
    Raises:
        ValueError: If date format is invalid
    """
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Date must be in format 'YYYY-M-D'")
    
def validate_timeframe(timeframe: str) -> str:
    """
    Validates and standardizes timeframe notation.
    
    Args:
        timeframe: Input timeframe (e.g., '1m', 'daily', 'weekly')
        
    Returns:
        Standardized timeframe notation
        
    Raises:
        ValueError: If timeframe is not supported
    """
    timeframe_map: Dict[str, str] = {
        # Minutes
        '1m':  '1m',
        '3m':  '3m',
        '5m':  '5m',
        '15m': '15m',
        '30m': '30m',
        
        # Hours
        '1h':  '1h',
        '2h':  '2h',
        '4h':  '4h',
        '6h':  '6h',
        '8h':  '8h',
        '12h': '12h',
        
        # Days
        'day':   '1d',
        'daily': '1d',
        '1d':    '1d',
        
        # Weeks
        'week':   '1w',
        'weekly': '1w',
        '1w':     '1w',
        
        # Months
        'month':   '1M',
        'monthly': '1M',
        '1M':      '1M'
    }
    
    timeframe = timeframe.lower()
    if timeframe in timeframe_map:
        return timeframe_map[timeframe]
    else:
        raise ValueError(f"Invalid timeframe. Available timeframes: {', '.join(timeframe_map.keys())}")
    
def fetch_historical_data(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """
    Fetches historical OHLCV data from specified exchange.
    
    Args:
        exchange: Initialized exchange instance
        symbol: Trading pair symbol
        timeframe: Validated timeframe string
        start_date: Start date for data fetch
        end_date: End date for data fetch
        
    Returns:
        OHLCV data with datetime index, or None if fetch fails
    """
    try:
        # Define millisecond conversion constants
        MINUTE_MS: int = 60 * 1000
        HOUR_MS: int = 60 * MINUTE_MS
        DAY_MS: int = 24 * HOUR_MS

        # Map timeframes to milliseconds for proper time handling
        timeframe_ms: Dict[str, int] = {
            '1m': MINUTE_MS,
            '3m': 3 * MINUTE_MS,
            '5m': 5 * MINUTE_MS,
            '15m': 15 * MINUTE_MS,
            '30m': 30 * MINUTE_MS,
            '1h': HOUR_MS,
            '2h': 2 * HOUR_MS,
            '4h': 4 * HOUR_MS,
            '6h': 6 * HOUR_MS,
            '8h': 8 * HOUR_MS,
            '12h': 12 * HOUR_MS,
            '1d': DAY_MS,
            '1w': 7 * DAY_MS,
            '1M': 30 * DAY_MS,
        }

        # Set exchange-specific limits
        limit: int = 200 if exchange.id == 'upbit' else 1000
        
        sleep_time: float = 0.05  # Rate limiting delay (50ms)
        all_ohlcv: List[List[Union[int, float]]] = []
        current_date: datetime = start_date
        
        print(f"{exchange.id}: 0%|", end='')
        
        while current_date <= end_date:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=int(current_date.timestamp() * 1000),
                    limit=limit
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # Update current_date based on the last candle timestamp
                last_candle_time = ohlcv[-1][0]
                current_date = datetime.fromtimestamp(last_candle_time / 1000) + timedelta(milliseconds=timeframe_ms[timeframe])
                
                # Progress indication
                progress = min(100, int((current_date - start_date) / (end_date - start_date) * 100))
                print(f"\r{exchange.id}: {progress}%|", end='')
                
                # Rate limiting
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Error fetching data: {str(e)}")
                break

        print()  # New line after progress

        if not all_ohlcv:
            print(f"No data returned from {exchange.id}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Filter to requested date range
        df = df.loc[start_date:end_date]
        
        # For timeframes less than daily, keep the full timestamp
        if timeframe != '1d' and timeframe != '1w' and timeframe != '1M':
            # Ensure the index maintains hours, minutes, seconds
            df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
            df.index = pd.to_datetime(df.index)
        else:
            # For daily/weekly/monthly data, use date format only
            df.index = df.index.date
            df.index = pd.to_datetime(df.index)

        return df

    except Exception as e:
        print(f"Error in fetch_historical_data: {str(e)}")
        return None
    
def find_suitable_cache(
    data_dir: str,
    exchange: str,
    pair: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """
    Searches for and returns cached data that matches the request parameters.
    
    Args:
        data_dir: Directory containing cached data files
        exchange: Exchange name
        pair: Trading pair
        timeframe: Data timeframe
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Cached data if found and valid, None otherwise
    """
    if not os.path.exists(data_dir):
        return None
        
    # Convert pair name for filename matching
    pair_str = pair.lower().replace('/', '2').replace(':', '')
    
    # List all pickle files for this exchange and timeframe
    cache_files = [f for f in os.listdir(data_dir) if f.startswith(f"{exchange}_{pair_str}_{timeframe}_")]
    
    for cache_file in cache_files:
        try:
            # Extract dates from filename
            parts = cache_file.split('_')
            cache_start = datetime.strptime(f"{parts[-6]}_{parts[-5]}_{parts[-4]}", '%Y_%m_%d')
            cache_end = datetime.strptime(f"{parts[-3]}_{parts[-2]}_{parts[-1].split('.')[0]}", '%Y_%m_%d')
            
            if cache_start <= start_date and cache_end >= end_date:
                cached_data = pd.read_pickle(os.path.join(data_dir, cache_file))
                filtered_data = cached_data[start_date:end_date]
                if not filtered_data.empty:
                    return filtered_data
                
        except Exception as e:
            print(f"Error reading cache file {cache_file}: {str(e)}")
            continue
            
    return None

def fetch_and_save_data(
    start_date: datetime,
    end_date: datetime,
    timeframe: str,
    base_data_dir: str = 'data'
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Main function to fetch and cache data from Binance and Upbit.
    
    Args:
        start_date: Start date for data fetch
        end_date: End date for data fetch
        timeframe: Data timeframe
        base_data_dir: Base directory for data storage
        
    Returns:
        Tuple containing DataFrames from both exchanges (can be None if fetch fails)
    """
    try:
        converted_timeframe = validate_timeframe(timeframe)
        
        abs_base_dir = os.path.abspath(base_data_dir)
        data_dir = os.path.join(abs_base_dir, converted_timeframe)
        os.makedirs(data_dir, exist_ok=True)

        # Hardcoded pairs
        binance_pair = 'BTC/USDT:USDT'  # Binance perpetual futures
        upbit_pair = 'BTC/KRW'          # Upbit spot

        start_date_str = start_date.strftime('%Y_%m_%d')
        end_date_str = end_date.strftime('%Y_%m_%d')

        binance_data = find_suitable_cache(data_dir, 'binance', binance_pair, converted_timeframe, start_date, end_date)
        upbit_data = find_suitable_cache(data_dir, 'upbit', upbit_pair, converted_timeframe, start_date, end_date)

        if binance_data is None or upbit_data is None:
            # Initialize exchanges
            binance = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}  # Set to futures market
            })
            upbit = ccxt.upbit({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })

            # Generate filenames
            binance_filename = f"binance_{binance_pair.lower().replace('/', '2').replace(':', '')}_{converted_timeframe}_{start_date_str}_{end_date_str}"
            upbit_filename = f"upbit_{upbit_pair.lower().replace('/', '2')}_{converted_timeframe}_{start_date_str}_{end_date_str}"
            
            binance_file = os.path.join(data_dir, f'{binance_filename}.pkl')
            upbit_file = os.path.join(data_dir, f'{upbit_filename}.pkl')

            if binance_data is None:
                print("\n🔄 Fetching new data from Binance Futures...")
                binance_data = fetch_historical_data(
                    binance,
                    binance_pair,
                    converted_timeframe,
                    start_date,
                    end_date
                )
                if binance_data is not None:
                    binance_data.to_pickle(binance_file)
                    print(f"✅ Binance data saved to: {binance_file}")

            if upbit_data is None:
                print("\n🔄 Fetching new data from Upbit...")
                upbit_data = fetch_historical_data(
                    upbit,
                    upbit_pair,
                    converted_timeframe,
                    start_date,
                    end_date
                )
                if upbit_data is not None:
                    upbit_data.to_pickle(upbit_file)
                    print(f"✅ Upbit data saved to: {upbit_file}")

        else:
            print("\n✅ Using cached data for both exchanges")

        if binance_data is not None and upbit_data is not None:
            print("\n📊 Data summary:")
            print(f"Binance rows: {len(binance_data)}")
            print(f"Upbit rows: {len(upbit_data)}")
            print(f"\n📅 Date ranges:")
            print(f"Binance: {binance_data.index.min()} to {binance_data.index.max()}")
            print(f"Upbit: {upbit_data.index.min()} to {upbit_data.index.max()}")

        return binance_data, upbit_data

    except Exception as e:
        print(f"Error in fetch_and_save_data: {str(e)}")
        return None, None

def parse_args() -> argparse.Namespace:
    """
    Parses command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Fetch cryptocurrency data from Binance and Upbit')
    parser.add_argument('-s', '--start_date', type=str, default='2021-1-1',
                        help='Start date (YYYY-MM-DD, E.g. 2021-1-1')
    parser.add_argument('-e', '--end_date', type=str, default='2023-1-1',
                        help='End date (YYYY-MM-DD), E.g. 2023-1-1')
    parser.add_argument('-t', '--timeframe', type=str, default='daily',
                        help='Timeframe (1m, 3m, 5m, daily, weekly, etc.), E.g. daily')
    parser.add_argument('-d', '--data_dir', type=str, default='data',
                        help='Directory to save data, E.g. data')

    parser.epilog = """
    Example usage:
    python fetch.py -s 2021-1-1 -e 2023-1-1 -t daily -d data
    """
    return parser.parse_args()

def fetch_data(
    start_date: str = '2021-1-1',
    end_date: str = '2023-1-1',
    timeframe: str = '1d'
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetch data from Binance and Upbit with given parameters.
    
    Args:
        start_date: Start date string in YYYY-MM-DD format
        end_date: End date string in YYYY-MM-DD format
        timeframe: Data timeframe
        
    Returns:
        DataFrames from both Binance and Upbit respectively
    """
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        print("Please use format YYYY-MM-DD (e.g., 2022-1-1 or 2022-01-01)")
        return None, None

    return fetch_and_save_data(
        start_date,
        end_date,
        timeframe
    )

def main() -> None:
    """
    Main entry point for command line usage.
    """
    args = parse_args()
    binance_data, upbit_data = fetch_data(
        args.start_date,
        args.end_date,
        args.timeframe
    )

if __name__ == '__main__':
    main()