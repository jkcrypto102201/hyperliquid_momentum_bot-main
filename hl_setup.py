# import os
# import sys
# import json
# import logging
# from decimal import Decimal
# from pathlib import Path
# from dotenv import load_dotenv
# from eth_account import Account
# from hyperliquid.info import Info
# from hyperliquid.exchange import Exchange


# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('hl_setup.log')
#     ]
# )
# logger = logging.getLogger(__name__)

# def setup():
#     """
#     Set up the Hyperliquid connection and account.
    
#     Returns:
#         tuple: (account_address, info, exchange)
#     """
#     # Load environment variables from .env file
#     load_dotenv()
    
#     # Get Hyperliquid keys
#     hl_private_key = os.getenv('HL_SECRET_KEY')
#     hl_account_address = os.getenv('HL_ACCOUNT_ADDRESS')
    
#     # Check if all required values are available
#     if not hl_private_key or not hl_account_address:
#         logger.error("HL_SECRET_KEY and HL_ACCOUNT_ADDRESS must be set in the .env file!")
#         raise ValueError("Missing Hyperliquid API credentials in .env file")
    
#     # Set up Hyperliquid account
#     account = Account.from_key(hl_private_key)
    
#     # Initialize API clients with mainnet
#     info = Info(base_url="https://api.hyperliquid.xyz")
#     exchange = Exchange(
#         base_url="https://api.hyperliquid.xyz", 
#         wallet=account,
#         account_address=hl_account_address
#     )
    
#     # Get account state for verification
#     try:
#         user_state = info.user_state(hl_account_address)
        
#         # Extract USDC equity (or total account value)
#         account_equity = None
#         if 'crossMarginSummary' in user_state and 'equity' in user_state['crossMarginSummary']:
#             account_equity = user_state['crossMarginSummary']['equity']
        
#         if account_equity is not None:
#             logger.info(f"Connected to Hyperliquid with account {hl_account_address}")
#             logger.info(f"Account equity: {account_equity} USDC")
#         else:
#             logger.warning(f"Connected to Hyperliquid, but could not retrieve account equity.")
#     except Exception as e:
#         logger.error(f"Error checking account state: {e}")
    
#     return hl_account_address, info, exchange

# def get_position_size(coin: str) -> Decimal:
#     """
#     Get current position size for a specific coin.
    
#     Args:
#         coin: The coin/asset to check position for
        
#     Returns:
#         Decimal: Position size (positive for long, negative for short, 0 for no position)
#     """
#     try:
#         # Load environment variables
#         load_dotenv()
#         hl_account_address = os.getenv('HL_ACCOUNT_ADDRESS')
        
#         if not hl_account_address:
#             logger.error("HL_ACCOUNT_ADDRESS must be set in the .env file!")
#             return Decimal('0')
        
#         # Initialize API client
#         info = Info(base_url="https://api.hyperliquid.xyz")
        
#         # Get user state
#         user_state = info.user_state(hl_account_address)
        
#         # Find position for the requested coin
#         if 'assetPositions' in user_state:
#             for position in user_state['assetPositions']:
#                 if position['coin'] == coin:
#                     # Position size with sign (positive for long, negative for short)
#                     return Decimal(position['position']['szi'])
        
#         # No position found
#         return Decimal('0')
    
#     except Exception as e:
#         logger.error(f"Error getting position size for {coin}: {e}")
#         return Decimal('0')

# # Initialize global variables on module import
# try:
#     account_address, info, exchange = setup()
# except Exception as e:
#     logger.error(f"Failed to initialize Hyperliquid connection: {e}")
#     # Keep references but they'll be None
#     account_address, info, exchange = None, None, None 













import os
import logging
from decimal import Decimal
from typing import Tuple, Optional
from dotenv import load_dotenv
from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from websocket import WebSocketTimeoutException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hl_setup.log')
    ]
)
logger = logging.getLogger(__name__)

def setup(base_url: Optional[str] = None, skip_ws: bool = False) -> Tuple[str, Info, Exchange]:
    """
    Set up the Hyperliquid connection and account for perpetual trading.
    """
    load_dotenv()
    
    # Get Hyperliquid keys
    hl_private_key = os.getenv('HL_SECRET_KEY')
    hl_account_address = os.getenv('HL_ACCOUNT_ADDRESS')
    
    if not hl_private_key:
        logger.error("HL_SECRET_KEY must be set in the .env file!")
        raise ValueError("Missing Hyperliquid secret key in .env file")
    
    # Initialize account
    account: LocalAccount = Account.from_key(hl_private_key)
    
    # Use wallet address if no account address provided
    if not hl_account_address:
        hl_account_address = account.address
        logger.info(f"No account address provided, using wallet address: {hl_account_address}")
    
    logger.info(f"Running with account address: {hl_account_address}")
    if hl_account_address != account.address:
        logger.info(f"Running with agent address: {account.address}")
    
    # Set default base URL if not provided
    if base_url is None:
        base_url = "https://api.hyperliquid.xyz"
    
    # Initialize API clients
    info = Info(base_url, skip_ws)
    exchange = Exchange(account, base_url, account_address=hl_account_address)
    
    # Verify connection and log balances
    try:
        info.meta()
        logger.info("Hyperliquid connection verified")
        
        # Get all balances
        user_state = info.user_state(hl_account_address)
        
        # 1. Check for USDC in margin summary
        if 'marginSummary' in user_state:
            margin_summary = user_state['marginSummary']
            logger.info(f"Margin summary: {margin_summary}")
            
            # This is where your USDC perps balance should appear
            if 'accountValue' in margin_summary:
                logger.info(f"Account value: {margin_summary['accountValue']} USDC")
        
        # 2. Check for existing positions
        if 'assetPositions' in user_state:
            for position in user_state['assetPositions']:
                coin = position['position']['coin']
                size = position['position']['szi']
                logger.info(f"Perps position: {size} {coin}")
        
        # 3. Check spot balances separately
        spot_user_state = info.spot_user_state(hl_account_address)
        if spot_user_state.get('balances'):
            for balance in spot_user_state['balances']:
                if float(balance['total']) > 0:
                    logger.info(f"Spot balance: {balance['total']} {balance['coin']}")
        
        # 4. Special check for wallet balances (includes perps collateral)
        wallet = info.wallet(hl_account_address)
        if wallet:
            logger.info(f"Wallet balances: {wallet}")
        
    except (ConnectionError, WebSocketTimeoutException) as e:
        logger.error(f"Connection test failed: {e}")
        raise
    except Exception as e:
        logger.warning(f"Could not fully verify account state: {e}")
    
    return hl_account_address, info, exchange


# def get_position_size(coin: str, info: Info, account_address: str) -> Decimal:
#     """Get current position size for a specific coin."""
#     try:
#         user_state = info.user_state(account_address)
#         if 'assetPositions' in user_state:
#             for position in user_state['assetPositions']:
#                 if position['position']['coin'] == coin:
#                     return Decimal(str(position['position']['szi']))
#         return Decimal('0')
#     except Exception as e:
#         logger.error(f"Error getting position size for {coin}: {e}")
#         return Decimal('0')
    

def get_position_size(coin: str) -> Decimal:
    """
    Get current position size for a specific coin.
    
    Args:
        coin: The coin/asset to check position for
        
    Returns:
        Decimal: Position size (positive for long, negative for short, 0 for no position)
    """
    try:
        # Load environment variables
        load_dotenv()
        hl_account_address = os.getenv('HL_ACCOUNT_ADDRESS')
        
        if not hl_account_address:
            logger.error("HL_ACCOUNT_ADDRESS must be set in the .env file!")
            return Decimal('0')
        
        # Initialize API client
        info = Info(base_url="https://api.hyperliquid.xyz")
        
        # Get user state
        user_state = info.user_state(hl_account_address)
        
        # Find position for the requested coin
        if 'assetPositions' in user_state:
            for position in user_state['assetPositions']:
                if position['coin'] == coin:
                    # Position size with sign (positive for long, negative for short)
                    return Decimal(position['position']['szi'])
        
        # No position found
        return Decimal('0')
    
    except Exception as e:
        logger.error(f"Error getting position size for {coin}: {e}")
        return Decimal('0')




# Initialize global variables
try:
    account_address, info, exchange = setup()
except Exception as e:
    logger.error(f"Failed to initialize Hyperliquid connection: {e}")
    raise