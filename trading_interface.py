import logging
import random
import time
from decimal import Decimal
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
from hyperliquid.info import Info
from hyperliquid.utils.types import Cloid


load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

@dataclass
class FillInfo:
    """Standardized fill information for trades."""
    timestamp: int  # Unix timestamp in milliseconds
    price: float
    size: float
    fee: float
    fee_currency: str
    side: str  # 'buy' or 'sell'
    is_maker: bool


@dataclass
class TradeResult:
    """Standardized trade result object."""
    success: bool
    ticker: str
    side: str
    intended_size: float
    filled_size: float
    average_price: Optional[float] = None
    total_fee: Optional[float] = None
    fills: Optional[List[FillInfo]] = None
    error_message: Optional[str] = None
    exchange_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    remaining_size: Optional[float] = None


class HyperliquidTrader():
    """Implementation of ExchangeTrader for Hyperliquid."""
    
    def __init__(self, exchange):
        """Initialize with a Hyperliquid exchange object."""
        self.exchange = exchange
    

    def place_limit_order(
        self, 
        ticker: str, 
        side: str, 
        size: float, 
        price: float,
        time_in_force: str = 'GTC',
        reduce_only: bool = False
    ) -> TradeResult:
        """
        Place a limit order on Hyperliquid.
        """
        try:
            # Convert parameters to Hyperliquid's expected format
            is_buy = side.lower() == 'buy'
            tif = 'Ioc' if time_in_force == 'IOC' else 'Gtc'

            # Get coin precision info
            precision = self.get_ticker_precision(ticker)
            size_decimals = precision['size_decimals']
            price_decimals = precision['price_decimals']

            # Calculate order value
            order_value = size * price 

            # Adjust size if order value is below $10
            if order_value < 10 and not reduce_only:
                adjusted_size = 10 / price
                # Round to the coin's required precision
                adjusted_size = round(adjusted_size, size_decimals)
                
                # Verify the adjusted size meets minimum after rounding
                if adjusted_size * price < 10:
                    # If still below $10, increment by minimum step
                    min_step = 10 ** (-size_decimals)
                    adjusted_size += min_step
                    adjusted_size = round(adjusted_size, size_decimals)

                logger.info(f"Adjusting size from {size} to {adjusted_size} to meet $10 minimum order value")
                size = adjusted_size

            # Format size and price with correct precision
            formatted_size = str(Decimal(str(size)).quantize(Decimal(f"1e-{size_decimals}")))
            formatted_price = str(Decimal(str(price)).quantize(Decimal(f"1e-{price_decimals}")))      

            # Generate 16 random bytes and convert to hex
            random_bytes = os.urandom(16)
            cloid_hex = random_bytes.hex()
            cloid_str = f"0x{cloid_hex}"

            # Create Cloid object
            cloid = Cloid.from_str(cloid_str)
            
            # Call the exchange.order() method with separate arguments
            response = self.exchange.order(
                ticker,
                is_buy,
                float(formatted_size),
                float(formatted_price),
                {"limit": {"tif": tif}},
                reduce_only,
                cloid
            )

            logger.info(f"Order response: {response}")
            
            # Initialize result with basic information
            result = TradeResult(
                success=True,
                ticker=ticker,
                side=side,
                intended_size=size,
                filled_size=0.0,
                fills=[],
                exchange_order_id=cloid
            )
            
            # Parse response
            if isinstance(response, dict):
                if 'status' in response:
                    status = response['status']
                    
                    if isinstance(status, dict):
                        if status.get('type') == 'filled' or status.get('ty') == 'filled':
                            # Order was filled immediately
                            if 'fills' in response:
                                total_fill_size = 0.0
                                total_fill_value = 0.0
                                total_fee = 0.0
                                
                                for fill in response['fills']:
                                    try:
                                        fill_price = float(fill['px'])
                                        fill_size = float(fill['sz'])
                                        fill_fee = float(fill['fee'])
                                        
                                        total_fill_size += fill_size
                                        total_fill_value += fill_price * fill_size
                                        total_fee += fill_fee
                                        
                                        fill_info = FillInfo(
                                            timestamp=int(time.time() * 1000),
                                            price=fill_price,
                                            size=fill_size,
                                            fee=fill_fee,
                                            fee_currency='USD',
                                            side=side,
                                            is_maker=False
                                        )
                                        result.fills.append(fill_info)
                                    except Exception as e:
                                        logger.error(f"Error processing fill data: {e}, fill: {fill}")
                                
                                if total_fill_size > 0:
                                    result.average_price = total_fill_value / total_fill_size
                                    result.filled_size = total_fill_size
                                    result.total_fee = total_fee
                                    result.remaining_size = size - total_fill_size
                                    result.exchange_order_id = str(response.get('oid', ''))
                                    
                            return result
                        elif status.get('type') == 'accepted' or status.get('ty') == 'accepted':
                            # Order was accepted but not filled yet
                            result.exchange_order_id = str(status.get('oid', ''))
                            result.filled_size = 0.0
                            result.remaining_size = size
                            return result
                        else:
                            # Order was not accepted
                            result.success = False
                            result.error_message = f"Order was not accepted: {response}"
                            logger.error(f"Order failed: {response}")
                            return result
                    elif isinstance(status, str) and status == 'ok':
                        # Successful order placement
                        if 'response' in response and isinstance(response['response'], dict):
                            if response['response'].get('type') == 'order':
                                result.exchange_order_id = str(response['response'].get('data', {}).get('statuses', [{}])[0].get('oid', ''))
                                return result
                    
                # Handle case where response has 'error' instead of 'status'
                if 'error' in response:
                    result.success = False
                    result.error_message = response['error']
                    return result
                
                # Handle unexpected response format
                result.success = False
                result.error_message = f"Unexpected response format: {response}"
                logger.error(f"Unexpected response: {response}")
                return result
            
            # Handle non-dictionary responses
            result.success = False
            result.error_message = f"Unexpected response type: {type(response)}"
            logger.error(f"Unexpected response type: {type(response)}")
            return result
        
        except Exception as e:
            error_msg = f"Error placing limit order: {e}"
            logger.error(error_msg, exc_info=True)
            return TradeResult(
                success=False,
                ticker=ticker,
                side=side,
                intended_size=size,
                filled_size=0.0,
                error_message=error_msg
            )

    def place_market_order(
        self, 
        ticker: str, 
        side: str, 
        size: float,
        reduce_only: bool = False
    ) -> TradeResult:
        """
        Place a market order on Hyperliquid.
        """
        try:
            # Get coin precision info
            precision = self.get_ticker_precision(ticker)
            size_decimals = precision['size_decimals']

            # Convert parameters to Hyperliquid's expected format
            is_buy = side.lower() == 'buy'
            
            # Format size with correct precision
            formatted_size = str(Decimal(str(size)).quantize(Decimal(f"1e-{size_decimals}")))

            # Generate 16 random bytes and convert to hex
            random_bytes = os.urandom(16)
            cloid_hex = random_bytes.hex()
            cloid_str = f"0x{cloid_hex}"

            # Create Cloid object
            cloid = Cloid.from_str(cloid_str)


            # Call the exchange.order() method with correct parameters
            response = self.exchange.order(
                ticker,
                is_buy,
                float(formatted_size),
                None,
                {"market": {}},
                reduce_only,
                cloid
            )
            
            # Initialize result with basic information
            result = TradeResult(
                success=True,
                ticker=ticker,
                side=side,
                intended_size=size,
                filled_size=0.0,
                fills=[],
                exchange_order_id=cloid
            )
            
            # Parse response
            if isinstance(response, dict):
                if 'status' in response:
                    status = response['status']
                    
                    if isinstance(status, dict):
                        if status.get('ty') == 'accepted':
                            result.exchange_order_id = status.get('oid')
                            
                            # Process fills if available
                            if 'fills' in response:
                                total_fill_size = 0.0
                                total_fill_value = 0.0
                                total_fee = 0.0
                                
                                for fill in response['fills']:
                                    try:
                                        fill_price = float(fill['px'])
                                        fill_size = float(fill['sz'])
                                        fill_fee = float(fill['fee'])
                                        
                                        total_fill_size += fill_size
                                        total_fill_value += fill_price * fill_size
                                        total_fee += fill_fee
                                        
                                        fill_info = FillInfo(
                                            timestamp=int(time.time() * 1000),
                                            price=fill_price,
                                            size=fill_size,
                                            fee=fill_fee,
                                            fee_currency='USD',
                                            side=side,
                                            is_maker=False
                                        )
                                        result.fills.append(fill_info)
                                    except Exception as e:
                                        logger.error(f"Error processing fill data: {e}, fill: {fill}")
                                
                                if total_fill_size > 0:
                                    result.average_price = total_fill_value / total_fill_size
                                
                                result.filled_size = total_fill_size
                                result.total_fee = total_fee
                                result.remaining_size = size - total_fill_size
                            return result
                        else:
                            result.success = False
                            result.error_message = f"Order was not accepted: {response}"
                            logger.error(f"Order failed: {response}")
                            return result
                    elif isinstance(status, str) and status == 'ok':
                        # Successful order placement
                        if 'response' in response and isinstance(response['response'], dict):
                            if response['response'].get('type') == 'order':
                                result.exchange_order_id = str(response['response'].get('data', {}).get('statuses', [{}])[0].get('oid', ''))
                                return result
                    
                # Handle case where response has 'error' instead of 'status'
                if 'error' in response:
                    result.success = False
                    result.error_message = response['error']
                    return result
                
                # Handle unexpected response format
                result.success = False
                result.error_message = f"Unexpected response format: {response}"
                logger.error(f"Unexpected response: {response}")
                return result
            
            # Handle non-dictionary responses
            result.success = False
            result.error_message = f"Unexpected response type: {type(response)}"
            logger.error(f"Unexpected response type: {type(response)}")
            return result
        
        except Exception as e:
            error_msg = f"Error placing market order: {e}"
            logger.error(error_msg)
            return TradeResult(
                success=False,
                ticker=ticker,
                side=side,
                intended_size=size,
                filled_size=0.0,
                error_message=error_msg
            )



    def get_ticker_precision(self, ticker: str) -> Dict[str, int]:
        """
        Get precision for ticker (decimals for price and size).
        """
        try:
            info = Info(base_url="https://api.hyperliquid.xyz")
            meta = info.meta()
            
            for coin_info in meta['universe']:
                if coin_info['name'] == ticker:
                    return {
                        # Note: szDecimals is for size, pxDecimals is for price
                        'size_decimals': int(coin_info.get('szDecimals', 2)),
                        'price_decimals': int(coin_info.get('pxDecimals', 8))
                    }
            
            return {
                'size_decimals': 2,
                'price_decimals': 8
            }
        except Exception as e:
            logger.error(f"Error getting ticker precision: {e}")
            return {
                'size_decimals': 2,
                'price_decimals': 8
            }



    def get_balance(self, currency: str = 'USD') -> Decimal:
        """
        Get balance for a specific currency.
        
        Args:
            currency: Currency to get balance for (only USD supported on Hyperliquid)
            
        Returns:
            Decimal balance
        """
        try:
            load_dotenv()
            account_address = os.getenv('HL_ACCOUNT_ADDRESS')
            if not account_address:
                return Decimal('0.0')
                
            info = Info(base_url="https://api.hyperliquid.xyz")
            # Initialize API client
            info = Info(base_url="https://api.hyperliquid.xyz")
            user_state = info.user_state(account_address)
            
            if 'crossMarginSummary' in user_state and 'accountValue' in user_state['crossMarginSummary']:
                return Decimal(str(user_state['crossMarginSummary']['accountValue']))
            return Decimal('0.0')
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return Decimal('0.0')


