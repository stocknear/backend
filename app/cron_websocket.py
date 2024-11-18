import asyncio
import websockets
import orjson
import os
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

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
                #logger.info(f"Processed data for {safe_symbol}")
        
        except orjson.JSONDecodeError:
            logger.warning(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def connect(self) -> None:
        """Establish WebSocket connection with auto-reconnect."""
        while True:
            try:
                async with websockets.connect(self.uri, ping_interval=30) as websocket:
                    # Login and subscribe
                    await websocket.send(orjson.dumps(self.login_payload))
                    await asyncio.sleep(2)
                    await websocket.send(orjson.dumps(self.subscribe_payload))
                    
                    # Handle incoming messages
                    async for message in websocket:
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