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

def check_market_hours() -> bool:
    """
    Check if the stock market is currently open.
    
    Returns:
        bool: True if market is open, False otherwise
    """
    # US stock market holidays for 2024
    us_holidays = [
        "2024-01-01",  # New Year's Day
        "2024-01-15",  # Martin Luther King Jr. Day
        "2024-02-19",  # Presidents' Day
        "2024-03-29",  # Good Friday
        "2024-05-27",  # Memorial Day
        "2024-06-19",  # Juneteenth
        "2024-07-04",  # Independence Day
        "2024-09-02",  # Labor Day
        "2024-11-28",  # Thanksgiving
        "2024-12-25",  # Christmas Day
    ]
    
    # Get current time in Eastern Time
    et_tz = zoneinfo.ZoneInfo('America/New_York')
    now = datetime.now(et_tz)
    # Check for weekend
    if now.weekday() >= 5:  # 5 and 6 are Saturday and Sunday
        return False
    
    # Check for holidays
    if now.strftime('%Y-%m-%d') in us_holidays:
        return False
    
    # Market hours are 9:30 AM to 4:00 PM ET
    market_open = time(9, 30)
    market_close = time(16, 0)
    current_time = now.time()
    
    # Check if current time is within market hours
    return market_open <= current_time < market_close

class WebSocketStockTicker:
    def __init__(self, api_key: str, uri: str = "wss://websockets.financialmodelingprep.com"):
        self.api_key = api_key
        self.uri = uri
        self.output_dir = Path('json/websocket/companies')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.login_payload = {
            "event": "login",
            "data": {"apiKey": self.api_key}
        }
        
        self.subscribe_payload = {
            "event": "subscribe",
            "data": {"ticker": ["*"]}
        }

    async def _safe_write(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Safely write data to file with error handling."""
        try:
            with open(file_path, 'wb') as f:
                f.write(orjson.dumps(data))
        except IOError as e:
            logger.error(f"File write error for {file_path}: {e}")

    async def _process_message(self, message: str) -> None:
        """Process and store individual WebSocket messages."""
        try:
            data = orjson.loads(message)
            
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
        """Establish WebSocket connection with auto-reconnect."""
        while True:
            # Check market hours before connecting
            if not check_market_hours():
                logger.info("Market is closed. Waiting 5 minutes before checking again.")
                await asyncio.sleep(300)  # Wait 5 minutes before checking again
                continue

            try:
                async with websockets.connect(self.uri, ping_interval=30) as websocket:
                    # Login and subscribe
                    await websocket.send(orjson.dumps(self.login_payload))
                    await asyncio.sleep(2)
                    await websocket.send(orjson.dumps(self.subscribe_payload))
                    
                    # Handle incoming messages
                    async for message in websocket:
                        # Additional check in case market closes during connection
                        if not check_market_hours():
                            logger.info("Market closed during connection. Disconnecting.")
                            break
                        await self._process_message(message)
            
            except (websockets.exceptions.ConnectionClosedError, 
                    websockets.exceptions.WebSocketException) as e:
                logger.warning(f"WebSocket error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

async def main():
    load_dotenv()
    api_key = os.getenv('FMP_API_KEY')
    
    if not api_key:
        logger.error("API Key not found. Please set FMP_API_KEY in .env file.")
        return
    
    ticker = WebSocketStockTicker(api_key)
    await ticker.connect()

if __name__ == "__main__":
    asyncio.run(main())