import time
import datetime
import pandas as pd
import numpy as np
import logging
from decimal import Decimal
from typing import List, Dict, Union, Optional
import sys
import os
import json
import signal
import sys

# Import local modules
from trading_interface import HyperliquidTrader
from hl_setup import info, exchange, get_position_size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_momentum.log')
    ]
)
logger = logging.getLogger(__name__)

# Define path to allocations file
ALLOCATIONS_FILE = os.path.join(os.path.dirname(__file__), 'mf_momentum_allocations.xlsx')

class MFMomentumStrategy:
    def __init__(
        self,
        timeframe: str,
        btc_timeframe: str,
        lookback_periods: int = 80,
        smooth_periods: int = None,
        entry_forecast_threshold: int = 10,
        exit_forecast_threshold: int = 5,
        exit_threshold: int = 5,
        btc_macd_filter: bool = True
    ):
        """
        Initialize the MF Momentum Strategy.
        
        Args:
            timeframe: Candle timeframe (e.g., "1m")
            lookback_periods: Number of periods to look back for min/max calculation
            smooth_periods: Number of periods for smoothing (defaults to lookback/4)
            forecast_threshold: Threshold for signal to enter a position (e.g., 10)
            exit_threshold: Threshold for signal to exit a position (e.g., 5)
            btc_macd_filter: Whether to use BTC MACD as a market filter
        """
        # Initialize with empty structures - will be populated from Excel file
        self.pairs = []
        self.notional_exposures = {}
        
        self.timeframe = timeframe
        self.btc_timeframe = btc_timeframe
        self.lookback_periods = lookback_periods
        self.smooth_periods = smooth_periods or max(int(lookback_periods / 4), 1)
        self.entry_forecast_threshold = entry_forecast_threshold
        self.exit_forecast_threshold = exit_forecast_threshold
        self.exit_threshold = exit_threshold
        self.btc_macd_filter = btc_macd_filter
        
        # Initialize trading interface
        self.hl_trader = HyperliquidTrader(exchange)
        
        # Initialize tick sizes as empty dictionary
        self.tick_sizes = {}
    
    def load_allocations(self) -> bool:
        """
        Load trading pairs and notional exposures from the allocations Excel file.
        
        Returns:
            bool: True if allocations were loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(ALLOCATIONS_FILE):
                logger.error(f"Allocations file not found at {ALLOCATIONS_FILE}. Please create this file with 'Ticker' and 'Notional' columns.")
                # If we don't have any pairs loaded yet, this is a fatal error
                if not self.pairs:
                    logger.critical("No allocation file exists and no default pairs are configured. Strategy cannot run.")
                return False
            
            # Read the Excel file
            df = pd.read_excel(ALLOCATIONS_FILE)
            logger.info(f"Loaded allocations file with {len(df)} entries")
            
            # Check if the file has the required columns
            required_columns = ['Ticker', 'Notional']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Allocations file is missing required columns. Needed: {required_columns}, Got: {df.columns.tolist()}")
                return False
            
            # Store the old pairs list for comparison
            old_pairs = set(self.pairs)
            
            # Update the pairs list and exposures
            new_pairs = []
            new_notional_exposures = {}
            
            for _, row in df.iterrows():
                ticker = row['Ticker']
                notional_size = row['Notional']
                
                # Skip rows with empty tickers or non-positive notional sizes
                if pd.isna(ticker) or pd.isna(notional_size) or notional_size <= 0:
                    continue
                
                # Add to new pairs and exposures
                new_pairs.append(ticker)
                new_notional_exposures[ticker] = float(notional_size)
            
            # Check if we got any valid pairs
            if not new_pairs:
                logger.error("No valid trading pairs found in allocations file")
                return False
                
            # Update the pairs list and notional exposures
            self.pairs = new_pairs
            self.notional_exposures = new_notional_exposures
            
            # Check if we have new pairs that need tick sizes
            new_pairs_set = set(new_pairs)
            added_pairs = new_pairs_set - old_pairs
            
            if added_pairs:
                logger.info(f"Found {len(added_pairs)} new pairs to add: {added_pairs}")
                # Update tick sizes for new pairs only
                self._update_tick_sizes_for_pairs(list(added_pairs))
            
            # Log the updated allocations
            allocations_log = {pair: self.notional_exposures.get(pair) for pair in self.pairs}
            logger.info(f"Updated allocations: {json.dumps(allocations_log, indent=2)}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading allocations: {e}")
            return False
    
    def _update_tick_sizes_for_pairs(self, pairs_to_update: List[str]) -> None:
        """
        Update tick sizes for specified pairs.
        
        Args:
            pairs_to_update: List of pairs to update tick sizes for
        """
        try:
            # Get metadata from Hyperliquid API
            meta = info.meta()
            
            # Initialize with default values
            for pair in pairs_to_update:
                self.tick_sizes[pair] = {
                    'price_decimals': 2,  # Default price decimals
                    'size_decimals': 8    # Default size decimals
                }
            
            # Extract size decimals from universe data
            if 'universe' in meta and isinstance(meta['universe'], list):
                for asset in meta['universe']:
                    if 'name' in asset and asset['name'] in pairs_to_update:
                        pair = asset['name']
                        # Size decimals are available directly in the API
                        if 'szDecimals' in asset:
                            self.tick_sizes[pair]['size_decimals'] = asset['szDecimals']
                            
                        # Log the asset data
                        logger.info(f"Asset data for new pair {pair}: {asset}")
            
            # For price decimals, analyze the order book
            for pair in pairs_to_update:
                try:
                    l2_snapshot = info.l2_snapshot(pair)
                    has_decimals = False
                    max_decimals = 0
                    
                    # Check both bids and asks sides
                    if 'levels' in l2_snapshot:
                        for side_idx in range(len(l2_snapshot['levels'])):
                            side = l2_snapshot['levels'][side_idx]
                            for level in side:
                                if 'px' in level:
                                    price = level['px']
                                    # Check if price has decimals that aren't just .0
                                    if '.' in price:
                                        decimal_part = price.split('.')[-1]
                                        # Only count if there are non-zero digits after decimal
                                        if decimal_part and any(d != '0' for d in decimal_part):
                                            has_decimals = True
                                            num_decimals = len(decimal_part)
                                            max_decimals = max(max_decimals, num_decimals)
                    
                    # If no meaningful decimals were found, set to 0
                    if not has_decimals:
                        self.tick_sizes[pair]['price_decimals'] = 0
                        logger.info(f"Determined price_decimals=0 for {pair} from order book (no meaningful decimals found)")
                    elif max_decimals > 0:
                        self.tick_sizes[pair]['price_decimals'] = max_decimals
                        logger.info(f"Determined price_decimals={max_decimals} for {pair} from order book")
                    
                except Exception as e:
                    logger.warning(f"Could not determine price decimals for {pair} from order book: {e}")
            
        except Exception as e:
            logger.error(f"Error updating tick sizes for new pairs: {e}")
    
    def _get_tick_sizes(self) -> Dict[str, Dict[str, int]]:
        """
        Get the tick sizes (precision) for all pairs in the strategy.
        
        Returns:
            Dictionary mapping tickers to their price and size precision
        """
        tick_sizes = {}
        
        try:
            # Get metadata from Hyperliquid API
            meta = info.meta()
            
            # Initialize with default values in case specific data retrieval fails
            for pair in self.pairs:
                tick_sizes[pair] = {
                    'price_decimals': 2,  # Default price decimals
                    'size_decimals': 8    # Default size decimals
                }
            
            # Extract size decimals from universe data
            if 'universe' in meta and isinstance(meta['universe'], list):
                for asset in meta['universe']:
                    if 'name' in asset and asset['name'] in self.pairs:
                        pair = asset['name']
                        # Size decimals are available directly in the API
                        if 'szDecimals' in asset:
                            tick_sizes[pair]['size_decimals'] = asset['szDecimals']
                            
                        # Log the asset data for debugging
                        logger.info(f"Asset data for {pair}: {asset}")
            
            # For price decimals, we need to analyze the order book
            for pair in self.pairs:
                try:
                    l2_snapshot = info.l2_snapshot(pair)
                    has_decimals = False
                    max_decimals = 0
                    
                    # Check both bids and asks sides
                    if 'levels' in l2_snapshot:
                        for side_idx in range(len(l2_snapshot['levels'])):
                            side = l2_snapshot['levels'][side_idx]
                            for level in side:
                                if 'px' in level:
                                    price = level['px']
                                    # Check if price has decimals that aren't just .0
                                    if '.' in price:
                                        decimal_part = price.split('.')[-1]
                                        # Only count if there are non-zero digits after decimal
                                        if decimal_part and any(d != '0' for d in decimal_part):
                                            has_decimals = True
                                            num_decimals = len(decimal_part)
                                            max_decimals = max(max_decimals, num_decimals)
                    
                    # If no meaningful decimals were found, set to 0
                    if not has_decimals:
                        tick_sizes[pair]['price_decimals'] = 0
                        logger.info(f"Determined price_decimals=0 for {pair} from order book (no meaningful decimals found)")
                    elif max_decimals > 0:
                        tick_sizes[pair]['price_decimals'] = max_decimals
                        logger.info(f"Determined price_decimals={max_decimals} for {pair} from order book")
                    
                except Exception as e:
                    logger.warning(f"Could not determine price decimals for {pair} from order book: {e}")
            
        except Exception as e:
            logger.error(f"Error getting tick sizes: {e}")
            # Use default values if an error occurred
        
        # Log the final tick sizes
        logger.info(f"Final tick sizes: {tick_sizes}")
        return tick_sizes
    
    def round_to_tick_size(self, pair: str, price: float) -> float:
        """
        Round a price to the appropriate tick size for the given pair.
        
        Args:
            pair: Trading pair
            price: Price to round
            
        Returns:
            Rounded price
        """
        if pair in self.tick_sizes:
            decimals = self.tick_sizes[pair]['price_decimals']
            return round(price, decimals)
        else:
            # Fallback to a simple decimal place calculation
            decimal_places = len(str(price).split('.')[-1]) if '.' in str(price) else 0
            rounding_factor = 10 ** decimal_places
            return round(price * rounding_factor) / rounding_factor
    
    def get_timeframe_seconds(self) -> int:
        """Convert timeframe string to seconds."""
        if self.timeframe.endswith('m'):
            return int(self.timeframe[:-1]) * 60
        elif self.timeframe.endswith('h'):
            return int(self.timeframe[:-1]) * 60 * 60
        elif self.timeframe.endswith('d'):
            return int(self.timeframe[:-1]) * 60 * 60 * 24
        else:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")
    
    def get_candles(self, ticker: str, limit: int = 1000, timeframe: str = None) -> pd.DataFrame:
        """
        Fetch historical candles from Hyperliquid.
        
        Args:
            ticker: Trading pair
            limit: Maximum number of candles to retrieve
            
        Returns:
            DataFrame with candle data (includes both open and close timestamps)
        """
        try:

            if timeframe is None:
                timeframe = self.timeframe

            # Calculate time range for candles based on requested limit and timeframe
            current_ts = int(time.time() * 1000)  # Current time in milliseconds (UTC)
            tf_seconds = self.get_timeframe_seconds()
            start_ts = current_ts - (limit * tf_seconds * 1000)  # Convert to milliseconds
            
            # Use candles_snapshot API for more precise control over timeframe
            candles_data = info.candles_snapshot(ticker, timeframe, 
                                              startTime=start_ts, endTime=current_ts)
            
            # Check if we got valid data
            if not candles_data:
                logger.error(f"No candle data returned for {ticker}")
                return pd.DataFrame()
            
            # Create DataFrame from candle data
            df = pd.DataFrame(candles_data)
            
            # Ensure we have the necessary columns
            required_columns = ['t', 'T', 'o', 'h', 'l', 'c', 'v']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Unexpected candle data format for {ticker}: {df.columns}")
                return pd.DataFrame()
            
            # Convert timestamps to datetime (UTC)
            df['open_time'] = pd.to_datetime(df['t'], unit='ms', utc=True)
            df['close_time'] = pd.to_datetime(df['T'], unit='ms', utc=True)
            
            # Keep the original timestamp column for backward compatibility
            df['timestamp'] = df['open_time']
            
            # Rename columns to standard OHLCV format
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            
            # Filter out the current unclosed candle
            # Current time in UTC milliseconds
            now_utc_ms = int(time.time() * 1000)
            # 'T' is the end time of the candle in UTC milliseconds
            df = df[df['T'] <= now_utc_ms]
            
            # Convert price and volume columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Select only the columns we need
            needed_columns = ['timestamp', 'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume']
            df = df[needed_columns]
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching candles for {ticker}: {e}")
            return pd.DataFrame()
    
    def compute_forecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the momentum forecast signal using the method described in the strategy.
        
        Args:
            df: DataFrame with candle data
            
        Returns:
            DataFrame with forecast signal
        """
        if df.empty:
            return df
        
        # Ensure numeric columns are float type
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Calculate rolling max and min
        roll_max = df["close"].rolling(window=self.lookback_periods, min_periods=self.lookback_periods//2).max()
        roll_min = df["close"].rolling(window=self.lookback_periods, min_periods=self.lookback_periods//2).min()
        
        # Calculate rolling mean
        roll_mean = (roll_max + roll_min) / 2
        
        # Calculate forecast
        forecast = 40.0 * (df["close"] - roll_mean) / (roll_max - roll_min)
        
        # Apply smoothing
        forecast = forecast.ewm(span=self.smooth_periods, min_periods=self.smooth_periods//2).mean()
        
        df["forecast"] = forecast
        
        return df.dropna()
    
    def compute_btc_macd(self) -> pd.DataFrame:
        """
        Compute BTC MACD for market filter.
        
        Returns:
            DataFrame with BTC MACD data
        """
        # Get BTC candles
        btc = self.get_candles("BTC", 1000, self.btc_timeframe)
        
        if btc.empty:
            logger.error("Failed to get BTC candles for MACD calculation")
            return pd.DataFrame()
        
        # Ensure numeric columns are float
        if 'close' in btc.columns:
            btc['close'] = btc['close'].astype(float)
        
        # Calculate MACD
        btc["macd"] = btc["close"].ewm(span=216).mean() - btc["close"].ewm(span=468).mean()
        btc["macd_signal"] = btc["macd"].ewm(span=162).mean()
        
        return btc[["timestamp", "macd", "macd_signal"]]
    
    def get_current_positions(self) -> Dict[str, int]:
        """
        Get current positions from Hyperliquid API.
        
        Returns:
            Dictionary mapping tickers to position sizes (1, 0, or -1)
        """
        positions = {}
        
        for pair in self.pairs:
            try:
                position_size = get_position_size(pair)
                
                if position_size > Decimal('0'):
                    positions[pair] = 1
                elif position_size < Decimal('0'):
                    positions[pair] = -1
                else:
                    positions[pair] = 0
                    
            except Exception as e:
                logger.error(f"Error getting position for {pair}: {e}")
                positions[pair] = 0
        
        return positions
    
    def get_next_candle_time(self) -> datetime.datetime:
        """
        Calculate the end time of the current candle.
        
        Returns:
            Datetime object representing the next candle time (in UTC)
        """
        # Get current time in UTC
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        seconds_in_timeframe = self.get_timeframe_seconds()
        
        # Calculate seconds since the start of the day (UTC)
        seconds_since_day_start = now_utc.hour * 3600 + now_utc.minute * 60 + now_utc.second
        
        # Calculate which candle we're in
        current_candle = seconds_since_day_start // seconds_in_timeframe
        
        # Calculate when the next candle starts
        next_candle_seconds = (current_candle + 1) * seconds_in_timeframe
        
        # Create datetime for the next candle (in UTC)
        next_candle_time = datetime.datetime.combine(
            now_utc.date(), 
            datetime.time(0),
            tzinfo=datetime.timezone.utc
        ) + datetime.timedelta(seconds=next_candle_seconds)
        
        # If we've passed midnight, adjust the date
        if next_candle_time < now_utc:
            next_candle_time += datetime.timedelta(days=1)
        
        return next_candle_time
    
    def compute_signal(self, pair: str, current_position: int) -> int:
        """
        Compute the trading signal for a pair.
        
        Args:
            pair: Trading pair
            
        Returns:
            Signal: 1 (long), 0 (neutral), or -1 (short)
        """
        # Get candles
        candles = self.get_candles(pair, 1000)  # Get enough candles for lookback

        if candles.empty:
            logger.error(f"Could not get candles for {pair}, returning neutral signal")
            return 0
        
        candles['ema20'] = candles['close'].ewm(span=20, min_periods=20).mean()
        candles['ema50'] = candles['close'].ewm(span=50, min_periods=50).mean()
        
        # Compute forecast
        candles_with_forecast = self.compute_forecast(candles)
        
        if candles_with_forecast.empty:
            logger.error(f"Could not compute forecast for {pair}, returning neutral signal")
            return 0
        
        # Get latest forecast
        latest_forecast = candles_with_forecast["forecast"].iloc[-1]

        latest_ema20 = candles_with_forecast["ema20"].iloc[-1]
        latest_ema50 = candles_with_forecast["ema50"].iloc[-1]
        
        # Get BTC MACD filter if enabled
        allow_long = True
        allow_short = True

        ema_filter = latest_ema20 > latest_ema50
        
        if self.btc_macd_filter:
            btc_macd = self.compute_btc_macd()
            
            if not btc_macd.empty:
                latest_btc = btc_macd.iloc[-1]
                allow_long = latest_btc["macd"] > latest_btc["macd_signal"]
                allow_short = latest_btc["macd"] < latest_btc["macd_signal"]

        
        if current_position == 0:
            if latest_forecast > self.entry_forecast_threshold and allow_long and ema_filter:
                return 1
            elif latest_forecast < -self.entry_forecast_threshold and allow_short and (not ema_filter):
                return -1
            else:
                return 0

        elif current_position == 1 and latest_forecast < self.exit_forecast_threshold:
            return 0
        elif current_position == -1 and latest_forecast > -self.exit_forecast_threshold:
            return 0
        else:
            return current_position
    
    def submit_order(self, pair: str, target_position: int, current_position: int) -> bool:
        """
        Submit an order to achieve the target position.
        Uses limit orders with IOC and 3% slippage for execution.
        
        Args:
            pair: Trading pair
            target_position: Target position (1, 0, or -1)
            current_position: Current position (1, 0, or -1)
            
        Returns:
            bool: True if order was successfully submitted
        """
        if target_position == current_position:
            return True  # No action needed
        
        # Get current price
        try:
            current_price = float(info.all_mids()[pair])
        except Exception as e:
            logger.error(f"Error getting price for {pair}: {e}")
            return False
        
        # Get the actual position size from the exchange if we're closing
        actual_position_size = 0
        if current_position != 0:  # If we have a position that needs to be closed
            try:
                actual_position_size = abs(float(get_position_size(pair)))
                if actual_position_size == 0:
                    logger.warning(f"Position size for {pair} is zero according to exchange, but expected {current_position}")
                    # If exchange says no position but we think there is one, update our state
                    current_position = 0
                else:
                    logger.info(f"Actual position size for {pair}: {actual_position_size}")
            except Exception as e:
                logger.error(f"Error getting actual position size for {pair}: {e}")
                # Fall back to calculated size
                notional_exposure = self.notional_exposures.get(pair, 0)
                actual_position_size = notional_exposure / current_price if notional_exposure > 0 else 0
                logger.warning(f"Using calculated position size: {actual_position_size}")
                
            # Round the size according to size decimals
            if pair in self.tick_sizes and actual_position_size > 0:
                size_decimals = self.tick_sizes[pair]['size_decimals']
                actual_position_size = round(actual_position_size, size_decimals)
                # Convert to string and back to ensure precision is exactly as needed
                size_str = f"{{:.{size_decimals}f}}".format(actual_position_size)
                actual_position_size = float(size_str)
        
        # CASE 1: Closing position completely (going to flat)
        if target_position == 0 and current_position != 0:
            close_side = 'sell' if current_position > 0 else 'buy'
            # Use more aggressive slippage (5%) for closing to ensure it gets filled
            close_limit_price = current_price * 0.95 if close_side == 'sell' else current_price * 1.05
            
            # Round the close price appropriately
            if pair in self.tick_sizes:
                price_decimals = self.tick_sizes[pair]['price_decimals']
                close_limit_price = round(close_limit_price, price_decimals)
            else:
                close_limit_price = self.round_to_tick_size(pair, close_limit_price)
                
            logger.info(f"Closing {current_position} position for {pair} with size {actual_position_size} at limit price {close_limit_price} (slippage: 5%)")
            
            result = self.hl_trader.place_limit_order(
                pair, 
                close_side, 
                actual_position_size, 
                close_limit_price,
                time_in_force='IOC',  # Immediate-or-Cancel
                reduce_only=True
            )
            
            if not result.success:
                logger.error(f"Failed to close position for {pair}: {result.error_message}")
                return False
            
            logger.info(f"Successfully closed {current_position} position for {pair} with limit order at {close_limit_price}")
            return True  # Successfully closed position, no need to open new one
            
        # CASE 2: For going from long to short or vice versa, we need to close position first
        elif current_position != 0 and target_position != 0 and target_position != current_position:
            # Close current position
            close_side = 'sell' if current_position > 0 else 'buy'
            # Use more aggressive slippage (5%) for closing to ensure it gets filled
            close_limit_price = current_price * 0.95 if close_side == 'sell' else current_price * 1.05
            
            # Round the close price appropriately
            if pair in self.tick_sizes:
                price_decimals = self.tick_sizes[pair]['price_decimals']
                close_limit_price = round(close_limit_price, price_decimals)
            else:
                close_limit_price = self.round_to_tick_size(pair, close_limit_price)
            
            logger.info(f"Closing {current_position} position for {pair} with size {actual_position_size} at limit price {close_limit_price} (slippage: 5%)")
            
            result = self.hl_trader.place_limit_order(
                pair, 
                close_side, 
                actual_position_size, 
                close_limit_price,
                time_in_force='IOC',  # Immediate-or-Cancel
                reduce_only=True
            )
            
            if not result.success:
                logger.error(f"Failed to close position for {pair}: {result.error_message}")
                return False
            
            logger.info(f"Closed {current_position} position for {pair} with limit order at {close_limit_price}")
        
        # CASE 3: Opening a new position (either from flat or after closing previous position)
        if target_position != 0:
            # Get notional exposure for this pair from the dictionary
            notional_exposure = self.notional_exposures.get(pair, 0)
            
            # If notional exposure is zero, we can't open a position
            if notional_exposure <= 0:
                logger.warning(f"Not opening position for {pair} as notional exposure is {notional_exposure}")
                return True
            
            # Calculate new position size
            new_position_size = notional_exposure / current_price
            
            # Round size according to size decimals
            if pair in self.tick_sizes:
                size_decimals = self.tick_sizes[pair]['size_decimals']
                new_position_size = round(new_position_size, size_decimals)
                # Convert to string and back to ensure precision is exactly as needed
                size_str = f"{{:.{size_decimals}f}}".format(new_position_size)
                new_position_size = float(size_str)
            
            # Determine order side
            side = 'buy' if target_position > 0 else 'sell'
            # Apply 3% slippage
            limit_price = current_price * 1.03 if side == 'buy' else current_price * 0.97
            
            # Round the price appropriately
            if pair in self.tick_sizes:
                price_decimals = self.tick_sizes[pair]['price_decimals']
                limit_price = round(limit_price, price_decimals)
            else:
                limit_price = self.round_to_tick_size(pair, limit_price)
                
            logger.info(f"Opening {target_position} position for {pair} with size {new_position_size} at limit price {limit_price} (slippage: 3%, notional: ${notional_exposure})")
            
            result = self.hl_trader.place_limit_order(
                pair, 
                side, 
                new_position_size, 
                limit_price,
                time_in_force='IOC',  # Immediate-or-Cancel
                reduce_only=False
            )
            
            if not result.success:
                logger.error(f"Failed to open position for {pair}: {result.error_message}")
                return False
            
            logger.info(f"Opened {target_position} position for {pair} with limit order at {limit_price} (filled at {result.average_price if result.average_price else 'unknown price'})")
        
        return True
    
    def run(self):
        """
        Run the momentum strategy continuously.
        """
        self._shutdown = False 

        while not self._shutdown:
            try:
                # Load allocations at the beginning of the run
                logger.info("Starting MF Momentum strategy - loading initial allocations from Excel file...")
                if not self.load_allocations():
                    logger.critical("Failed to load initial allocations. Please ensure the Excel file exists and has the correct format.")
                    logger.info(f"Expected file path: {ALLOCATIONS_FILE}")
                    logger.info("Required columns: 'Ticker', 'Notional'")
                    return
                    
                # Initialize tick sizes for all pairs
                if not self.tick_sizes:
                    self.tick_sizes = self._get_tick_sizes()
                    logger.info(f"Initialized tick sizes: {self.tick_sizes}")
                    
                logger.info(f"Starting with {len(self.pairs)} pairs")
                logger.info(f"Timeframe: {self.timeframe}")
                
                while True:
                    try:
                        # Calculate time until next candle (in UTC)
                        next_candle_time = self.get_next_candle_time()
                        now_utc = datetime.datetime.now(datetime.timezone.utc)
                        
                        # Calculate seconds to sleep - no additional delay
                        seconds_to_sleep = (next_candle_time - now_utc).total_seconds()
                        
                        if seconds_to_sleep > 0:
                            logger.info(f"Sleeping for {seconds_to_sleep:.2f} seconds until next candle at {next_candle_time} UTC")
                            time.sleep(seconds_to_sleep)
                        
                        # Reload allocations BEFORE trading at the end of the candle
                        logger.info("Reloading token allocations from Excel file...")
                        if not self.load_allocations():
                            logger.error("Failed to reload allocations, continuing with previous settings")
                        
                        # Get current positions
                        current_positions = self.get_current_positions()
                        logger.info(f"Current positions: {current_positions}")
                        
                        # Process each pair
                        for pair in self.pairs:
                            try:
                                # Skip pairs with zero notional exposure
                                if self.notional_exposures.get(pair, 0) <= 0:
                                    logger.info(f"Skipping {pair} as it has zero or negative notional exposure")
                                    # If we have a position but notional is now 0, close it
                                    if current_positions.get(pair, 0) != 0:
                                        logger.info(f"Closing position for {pair} as notional is now 0")
                                        self.submit_order(pair, 0, current_positions.get(pair, 0))
                                    continue
                                
                                # Compute signal
                                signal = self.compute_signal(pair, current_positions.get(pair, 0))
                                logger.info(f"Signal for {pair}: {signal}")
                                
                                # Submit order if signal is different from current position
                                if signal != current_positions.get(pair, 0):
                                    logger.info(f"Position change for {pair}: {current_positions.get(pair, 0)} -> {signal}")
                                    self.submit_order(pair, signal, current_positions.get(pair, 0))
                            
                            except Exception as e:
                                logger.error(f"Error processing pair {pair}: {e}")
                        
                        # Also handle closed positions for pairs that are no longer in the allocation list
                        removed_pairs = set(current_positions.keys()) - set(self.pairs)
                        for pair in removed_pairs:
                            if current_positions.get(pair, 0) != 0:
                                logger.info(f"Closing position for {pair} as it was removed from allocations")
                                self.submit_order(pair, 0, current_positions.get(pair, 0))
                        
                        # Sleep for a short time to avoid excessive API calls
                        time.sleep(5)
                        
                    except Exception as e:
                        logger.error(f"Strategy error: {e}")
                        time.sleep(30)  # Sleep and retry

            except KeyboardInterrupt:
                self._shutdown = True
                logger.info("Shutdown requested, finishing current operations...")



def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nShutting down gracefully...")
    sys.exit(0)



def main():
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Only timeframe is required, all other parameters will be loaded from Excel
    timeframe = "1m"
    btc_filter_timeframe = "1m"
    
    # Initialize and run strategy
    strategy = MFMomentumStrategy(
        timeframe=timeframe,
        btc_timeframe=btc_filter_timeframe,
        lookback_periods=80,
        entry_forecast_threshold=10,
        exit_forecast_threshold=5,
        exit_threshold=5,
        btc_macd_filter=True
    )
    
    try:
        strategy.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 

    