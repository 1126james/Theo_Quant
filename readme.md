# Kimchi Premium Trading Strategy Backtester

A Python-based backtesting framework for analyzing and optimizing trading strategies based on the Kimchi premium (price differences between Korean and international cryptocurrency markets).

## Features

- Automated data fetching from Upbit and Binance exchanges
- Parameter optimization using Scikit-Optimize
- Split testing (backtest and forward test) for strategy validation
- Performance visualization with heatmaps and equity curves
- Comprehensive performance metrics including CAGR, Sharpe ratio, and drawdown analysis

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation
pip install -r requirements.txt

## Usage
- Basic Usage
- Run the strategy with default parameters:

@ bash
python kimchi_strategy.py

# Custom Parameters
- Specify custom date ranges and timeframes:

@ bash
python kimchi_strategy.py --start_date 2021-01-01 --end_date 2023-01-31 --timeframe 1d

- Available Parameters
--start_date: Start date for analysis (format: YYYY-MM-DD)
--end_date: End date for analysis (format: YYYY-MM-DD)
--timeframe: Data timeframe (options: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d)

# Using as a Module
You can also import and use the strategy in your own Python code:

@ python
from kimchi_strategy import main

# Run with custom parameters
main(start_date='2021-01-01', 
     end_date='2023-01-31', 
     timeframe='1d')

# Strategy Logic
The strategy implements a cross-exchange arbitrage approach:

Monitors price movements in the source market (Upbit)
Opens positions in the target market (Binance) based on threshold crossings
Uses optimized threshold parameters for entry and exit decisions

# Key Parameters
x_threshold: Upward price movement threshold
y_threshold: Downward price movement threshold

# Output
- The strategy outputs:

1. Optimization results including:
2. Best parameter combinations
3. Parameter optimization heatmap
4. Equity curves for both backtest and forward test periods

# Performance metrics including:
- CAGR (Compound Annual Growth Rate)
- Return percentages
- Maximum drawdown
- Sharpe ratio
- Win rate

# Visualization
The strategy generates two main visualizations:

- Parameter Optimization Heatmap
- Shows the Sharpe ratio for different parameter combinations
- Highlights the optimal parameter combination
- Equity Curves
- Displays portfolio value over time
- Separates backtest and forward test periods



### Example Output

📊 Strategy Performance Summary:

Optimal Parameters:
X% (Upbit upward): 2.45%
Y% (Upbit downward): 1.87%

Backtest Period:
CAGR: 24.56%
Return: 45.67%
Maximum Drawdown: 15.43%
Sharpe Ratio: 1.89
Win Rate: 62.34%

Forward Test Period:
CAGR: 18.90%
Return: 32.45%
Maximum Drawdown: 18.76%
Sharpe Ratio: 1.45
Win Rate: 58.90%

# Error Handling
The code includes error handling for:

Invalid date formats
Insufficient data
API connection issues
Parameter optimization failures