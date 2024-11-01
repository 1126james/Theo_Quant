from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fetch import fetch_data
import skopt
import argparse
from datetime import datetime

class UpbitBinanceStrategy(Strategy):
    """
    A trading strategy class that implements a cross-exchange arbitrage strategy
    between Upbit and Binance markets.
    
    The strategy opens positions based on percentage changes in the source market
    and closes them when opposite thresholds are met.
    """
    x_threshold = 1  # Upward threshold for Upbit price movement
    y_threshold = 1  # Downward threshold for Upbit price movement
    
    def init(self):
        """Initialize strategy by creating indicators."""
        self.source_returns = self.I(lambda: self.data.source_returns)
    
    def next(self):
        """
        Define trading logic for each step.
        Implements position entry and exit based on source market returns.
        """
        if self.position and self.position.is_long:
            if self.source_returns[-1] <= -self.y_threshold:
                self.position.close()
        
        elif self.position and self.position.is_short:
            if self.source_returns[-1] >= self.x_threshold:
                self.position.close()
        
        elif not self.position:
            if self.source_returns[-1] >= self.x_threshold:
                self.buy()
            elif self.source_returns[-1] <= -self.y_threshold:
                self.sell()

def calculate_cagr(initial_value: float, final_value: float, n_years: float) -> float:
    """Calculate Compound Annual Growth Rate"""
    if initial_value <= 0 or n_years <= 0:
        return 0.0
    return (((final_value / initial_value) ** (1 / n_years)) - 1) * 100

def prepare_data_for_backtesting(df_source: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and format data for backtesting.
    
    Args:
        df_source: Source exchange data (Upbit)
        df_target: Target exchange data (Binance)
    
    Returns:
        Formatted DataFrame ready for backtesting
    """
    df_source = df_source.copy()
    df_target = df_target.copy()
    
    source_returns = df_source['close'].pct_change() * 100
    
    bt_data = pd.DataFrame({
        'Open': df_target['open'],
        'High': df_target['high'],
        'Low': df_target['low'],
        'Close': df_target['close'],
        'Volume': df_target['volume']
    })
    
    bt_data['source_returns'] = source_returns.reindex(bt_data.index)
    bt_data = bt_data.fillna(0)
    return bt_data

def optimize_and_test(df_source: pd.DataFrame, df_target: pd.DataFrame) -> dict:
    """
    Optimize strategy parameters and perform backtesting and forward testing.
    
    Args:
        df_source: Source exchange data
        df_target: Target exchange data
    
    Returns:
        Dictionary containing optimization results and performance metrics
    """
    bt_data = prepare_data_for_backtesting(df_source, df_target)
    
    # Split data into backtest and forward test periods
    dates = bt_data.index
    mid_point = dates[len(dates)//2]
    backtest_data = bt_data[bt_data.index < mid_point].copy()
    forward_test_data = bt_data[bt_data.index >= mid_point].copy()
    
    if len(backtest_data) == 0 or len(forward_test_data) == 0:
        raise ValueError("Not enough data for both backtesting and forward testing.")
    
    # Initialize backtest instance
    bt = Backtest(
        backtest_data,
        UpbitBinanceStrategy,
        cash=1_000_000_000,
        commission=0.001,
        margin=1.0,
        hedging=True,
        exclusive_orders=True
    )
    
    # Optimize strategy parameters
    stats, heatmap, optimize_result = bt.optimize(
        x_threshold=np.arange(0.1, 10.1, 0.1),
        y_threshold=np.arange(0.1, 10.1, 0.1),
        maximize='Sharpe Ratio',
        method='skopt',
        return_heatmap=True,
        return_optimization=True
    )
    
    best_params = {
        'x_threshold': optimize_result.x[0],
        'y_threshold': optimize_result.x[1]
    }
    
    # Run backtest with optimized parameters
    backtest_results = bt.run(**best_params)
    
    # Forward test with optimized parameters
    bt_forward = Backtest(
        forward_test_data,
        UpbitBinanceStrategy,
        cash=1_000_000_000,
        commission=0.001,
        margin=1.0,
        hedging=True,
        exclusive_orders=True
    )
    forward_test_results = bt_forward.run(**best_params)

    # Calculate performance metrics
    backtest_years = (backtest_data.index[-1] - backtest_data.index[0]).days / 365.25
    forward_test_years = (forward_test_data.index[-1] - forward_test_data.index[0]).days / 365.25
    
    backtest_cagr = calculate_cagr(1_000_000_000, backtest_results._equity_curve['Equity'].iloc[-1], backtest_years)
    forward_test_cagr = calculate_cagr(1_000_000_000, forward_test_results._equity_curve['Equity'].iloc[-1], forward_test_years)

    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot heatmap
    plt.subplot(2, 1, 1)
    if heatmap is not None:
        x_vals = sorted(set(idx[0] for idx in heatmap.index))
        y_vals = sorted(set(idx[1] for idx in heatmap.index))
        heatmap_matrix = np.zeros((len(x_vals), len(y_vals)))
        
        x_map = {x: i for i, x in enumerate(x_vals)}
        y_map = {y: i for i, y in enumerate(y_vals)}
        
        for (x, y), value in heatmap.items():
            heatmap_matrix[x_map[x], y_map[y]] = value
            
        plt.gca().set_facecolor('navy')
        im = plt.imshow(heatmap_matrix, cmap='RdYlGn', aspect='auto', origin='lower',
                       extent=[min(y_vals), max(y_vals), min(x_vals), max(x_vals)],
                       interpolation='gaussian')
        plt.colorbar(im, label='Sharpe Ratio')
        plt.title('Parameter Optimization Heatmap')
        plt.xlabel('Y% (Down Threshold)')
        plt.ylabel('X% (Up Threshold)')

        best_x, best_y = best_params['x_threshold'], best_params['y_threshold']
        plt.axhline(y=best_x, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=best_y, color='red', linestyle='--', alpha=0.5)
        plt.plot([best_y], [best_x], 'ro', markersize=8, label='Optimal Parameters')
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    # Plot equity curves
    plt.subplot(2, 1, 2)
    backtest_equity = backtest_results._equity_curve['Equity']
    forward_test_equity = forward_test_results._equity_curve['Equity']
    
    plt.plot(backtest_equity.index, backtest_equity, label='Backtest Period', alpha=0.7)
    plt.plot(forward_test_equity.index, forward_test_equity, label='Forward Test Period', alpha=0.7)
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return {
        'best_parameters': best_params,
        'optimization_heatmap': heatmap,
        'backtest_results': {
            'return': backtest_results['Return [%]'],
            'sharpe': backtest_results['Sharpe Ratio'],
            'max_drawdown': backtest_results['Max. Drawdown [%]'],
            'cagr': backtest_cagr,
            'n_trades': backtest_results['# Trades'],
            'win_rate': backtest_results['Win Rate [%]'],
            'profit_factor': backtest_results.get('Profit Factor', 0),
            'expectancy': backtest_results.get('Expectancy [%]', 0)
        },
        'forward_test_results': {
            'return': forward_test_results['Return [%]'],
            'sharpe': forward_test_results['Sharpe Ratio'],
            'max_drawdown': forward_test_results['Max. Drawdown [%]'],
            'cagr': forward_test_cagr,
            'n_trades': forward_test_results['# Trades'],
            'win_rate': forward_test_results['Win Rate [%]'],
            'profit_factor': forward_test_results.get('Profit Factor', 0),
            'expectancy': forward_test_results.get('Expectancy [%]', 0)
        }
    }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Kimchi Premium Trading Strategy Backtester')
    parser.add_argument('-s', '--start_date', type=str, default='2021-1-1',
                       help='Start date in YYYY-MM-DD format (default: 2021-1-1)')
    parser.add_argument('-e', '--end_date', type=str, default='2023-1-31',
                       help='End date in YYYY-MM-DD format (default: 2023-1-31)')
    parser.add_argument('-t', '--timeframe', type=str, default='1d',
                       choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'],
                       help='Timeframe for the data (default: 1d)')
    return parser.parse_args()

def main(start_date=None, end_date=None, timeframe=None):
    """
    Main function to run the strategy optimization and testing.
    
    Args:
        start_date: Start date for data fetching (YYYY-MM-DD)
        end_date: End date for data fetching (YYYY-MM-DD)
        timeframe: Data timeframe (e.g., '1d', '1h')
    """
    if all(v is None for v in [start_date, end_date, timeframe]):
        args = parse_args()
        start_date = args.start_date
        end_date = args.end_date
        timeframe = args.timeframe
    
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"Error: Invalid date format. Please use YYYY-MM-DD format. {str(e)}")
        return
    
    df_binance, df_upbit = fetch_data(start_date, end_date, timeframe)
    
    if df_upbit is None or df_binance is None or len(df_upbit) < 2 or len(df_binance) < 2:
        print("Error: Insufficient data for analysis")
        return
    
    try:
        results = optimize_and_test(df_upbit, df_binance)
        
        print("\n📊 Strategy Performance Summary:")
        print(f"\nOptimal Parameters:")
        print(f"X% (Upbit upward): {results['best_parameters']['x_threshold']:.2f}%")
        print(f"Y% (Upbit downward): {results['best_parameters']['y_threshold']:.2f}%")
        
        print("\nBacktest Period:")
        print(f"CAGR: {results['backtest_results']['cagr']:.2f}%")
        print(f"Return: {results['backtest_results']['return']:.2f}%")
        print(f"Maximum Drawdown: {results['backtest_results']['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['backtest_results']['sharpe']:.2f}")
        print(f"Win Rate: {results['backtest_results']['win_rate']:.2f}%")
        
        print("\nForward Test Period:")
        print(f"CAGR: {results['forward_test_results']['cagr']:.2f}%")
        print(f"Return: {results['forward_test_results']['return']:.2f}%")
        print(f"Maximum Drawdown: {results['forward_test_results']['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['forward_test_results']['sharpe']:.2f}")
        print(f"Win Rate: {results['forward_test_results']['win_rate']:.2f}%")
    
    except Exception as e:
        print(f"Error during optimization and testing: {str(e)}")

if __name__ == "__main__":
    main()