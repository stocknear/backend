import asyncio
import websockets
import orjson
import os
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from datetime import datetime, time
import zoneinfo
import aiofiles
import functools

# Use uvloop for faster event loop if available
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

# Optimize logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Precompute holidays and use a set for faster lookups
US_HOLIDAYS = {'2025-01-01', '2025-01-09','2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'}

# Use functools.cache to memoize market hours check
@functools.cache
def check_market_hours() -> bool:
    """
    Check if the stock market is currently open.
    
    Returns:
        bool: True if market is open, False otherwise
    """
    et_tz = zoneinfo.ZoneInfo('America/New_York')
    now = datetime.now(et_tz)
    
    # Quick weekend check
    if now.weekday() >= 5:
        return False
    
    # Use set for faster holiday lookup
    if now.strftime('%Y-%m-%d') in US_HOLIDAYS:
        return False
    
    # Market hours check
    current_time = now.time()
    return time(9, 30) <= current_time < time(16, 0)

class WebSocketStockTicker:
    def __init__(self, api_key: str, uri: str = "wss://websockets.financialmodelingprep.com"):
        # Use slots to reduce memory overhead
        __slots__ = ['api_key', 'uri', 'output_dir', 'login_payload', 'subscribe_payload']
        
        self.api_key = api_key
        self.uri = uri
        self.output_dir = Path('json/websocket/companies')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Precompute payloads to avoid repeated dictionary creation
        self.login_payload = orjson.dumps({
            "event": "login",
            "data": {"apiKey": self.api_key}
        })
        
        self.subscribe_payload = orjson.dumps({
            "event": "subscribe",
            "data": {"ticker": ["*"]}
        })


    async def _safe_write(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Safely write data to file using aiofiles for non-blocking I/O."""
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(orjson.dumps(data))
        except IOError as e:
            logger.error(f"File write error for {file_path}: {e}")

    async def _process_message(self, message: str) -> None:
        """Optimized message processing with minimal allocation."""
        try:
            data = orjson.loads(message)
            
            # Fast symbol extraction and sanitization
            if 's' in data:
                symbol = data['s'].upper()
                safe_symbol = ''.join(c for c in symbol if c.isalnum() or c in ['-', '_'])
                file_path = self.output_dir / f"{safe_symbol}.json"
                
                await self._safe_write(file_path, data)
        
        except orjson.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def connect(self) -> None:
        """Establish WebSocket connection with robust error handling."""
        reconnect_delay = 5
        max_reconnect_delay = 60
        
        while True:
            # Check market hours before connecting
            if not check_market_hours():
                logger.info("Market is closed. Waiting 5 minutes before checking again.")
                await asyncio.sleep(300)  # Wait 5 minutes
                continue

            try:
                async with websockets.connect(self.uri, ping_interval=30) as websocket:
                    # Reset reconnect delay on successful connection
                    reconnect_delay = 5
                    
                    # Login and subscribe with pre-serialized payloads
                    await websocket.send(self.login_payload)
                    await asyncio.sleep(2)
                    await websocket.send(self.subscribe_payload)
                    
                    # Handle incoming messages with timeout
                    async for message in websocket:
                        if not check_market_hours():
                            logger.info("Market closed during connection. Disconnecting.")
                            break
                        
                        # Use asyncio.create_task for concurrent message processing
                        asyncio.create_task(self._process_message(message))
            
            except (websockets.exceptions.ConnectionClosedError, 
                    websockets.exceptions.WebSocketException) as e:
                logger.warning(f"WebSocket error: {e}. Reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
                # Exponential backoff with cap
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
            except Exception as e:
                logger.error(f"Unexpected error: {e}. Reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

async def main():
    load_dotenv()
    api_key = os.getenv('FMP_API_KEY')
    
    ticker = WebSocketStockTicker(api_key)
    await ticker.connect()

if __name__ == "__main__":
    asyncio.run(main())