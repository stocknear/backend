import os
import json
import sqlite3
import concurrent.futures
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Configuration
DEFAULT_BASE_DIR = "../"
MAX_WORKERS = 5  # Adjust based on your API rate limits and system resources

class VectorStoreUploader:
    def __init__(self, symbol, data_type="historical-price", timeframe="max", base_dir=DEFAULT_BASE_DIR):
        """
        Initialize the uploader with configuration for a specific symbol and data type.
        
        Args:
            symbol: The stock/ETF/index symbol (e.g., 'GME')
            data_type: Type of data (e.g., 'historical-price', 'similar-stocks')
            timeframe: Data timeframe (e.g., 'max', '1y', '3mo')
            base_dir: Base directory for all data
        """
        self.symbol = symbol
        self.data_type = data_type
        self.timeframe = timeframe
        self.base_dir = Path(base_dir)
        
        # Set up OpenAI client
        self.vector_store_id = os.getenv("VECTOR_STORE_ID")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Configure paths based on data type
        if data_type == "similar-stocks":
            # For similar-stocks data, we use a simplified path structure without timeframe
            self.input_path = self.base_dir / f"json/{data_type}/{symbol}.json"
            self.state_path = self.base_dir / f"json/vector-store/{data_type}/{symbol}.json"
            self.vector_store_name = f"{symbol} Similar Stocks"
        else:
            # For historical price data and other time series
            self.input_path = self.base_dir / f"json/{data_type}/{timeframe}/{symbol}.json"
            self.state_path = self.base_dir / f"json/vector-store/{data_type}/{timeframe}/{symbol}.json"
            self.vector_store_name = f"{symbol} {data_type.replace('-', ' ').title()} ({timeframe})"
    
    def get_last_uploaded_date(self):
        """Read the last-uploaded ISO date from state_path, or return epoch if missing."""
        if os.path.exists(self.state_path):
            with open(self.state_path, "r") as f:
                return datetime.fromisoformat(f.read().strip())
        else:
            # If first run, use a very old date so you upload everything
            return datetime.fromisoformat("1970-01-01T00:00:00")
    
    def set_last_uploaded_date(self, dt: datetime):
        """Write the new last-uploaded ISO date to state_path, creating directories if needed."""
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w") as f:
            f.write(dt.isoformat())
    
    def load_new_entries(self, last_dt: datetime, date_field="time"):
        """Load the JSON file and return only entries newer than last_dt."""
        if not os.path.exists(self.input_path):
            logger.warning(f"Input file not found at {self.input_path}")
            return []
            
        with open(self.input_path, "r") as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            # If data is a dictionary, extract values that might be the records
            if "data" in data:
                records = data["data"]
            else:
                # If no "data" key, use all values that are lists
                records = next((v for v in data.values() if isinstance(v, list)), [])
        else:
            records = data
        
        # Check if records have the specified date field
        has_date_field = any(date_field in rec for rec in records) if records else False
        
        if has_date_field:
            # Filter by date field if it exists
            new_records = [
                rec for rec in records
                if date_field in rec and datetime.fromisoformat(rec[date_field]) > last_dt
            ]
        else:
            # If no date field exists in records, include all records and assign today's date
            today = datetime.now().replace(microsecond=0)
            new_records = []
            
            # Only upload if we haven't uploaded today
            if last_dt.date() < today.date():
                # Add today's date to each record for tracking purposes
                for rec in records:
                    # Create a shallow copy and add the date field
                    updated_rec = rec.copy()
                    updated_rec[date_field] = today.isoformat()
                    new_records.append(updated_rec)
        
        return new_records
    
    def get_or_create_store_id(self):
        """Get existing or create new vector store ID."""
        if self.vector_store_id:
            # If vector_store_id is already set, use it directly
            return self.vector_store_id
            
        resp = self.client.vector_stores.list()
        for vs in resp.data:
            if vs.name == self.vector_store_name:
                logger.info(f"Found Existing Vector Store: {self.vector_store_name}")
                return vs.id
                
        logger.info(f"No Vector Store available. Creating a new one: {self.vector_store_name}")
        new_vs = self.client.vector_stores.create(name=self.vector_store_name)
        return new_vs.id
    
    def upload_slice(self, vector_store_id: str, slice_records: list):
        """Upload the filtered slice to the vector store as one batch."""
        if not slice_records:
            logger.info(f"No new records to upload for {self.symbol}.")
            return None
            
        # Write slice to a temp file
        tmp_path = f"{self.input_path}.slice.json"
        with open(tmp_path, "w") as f:
            json.dump(slice_records, f)
            
        try:
            with open(tmp_path, "rb") as f:
                batch = self.client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store_id,
                    files=[f],
                )
                
            return batch
        finally:
            # Make sure we clean up the temp file, even if there's an error
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def process(self, date_field="time"):
        """Process a single symbol's data upload."""
        try:
            vector_store_id = self.get_or_create_store_id()
            last_dt = self.get_last_uploaded_date()
            
            logger.info(f"Processing {self.symbol} from {self.data_type} ({self.timeframe})")
            logger.info(f"Last uploaded date: {last_dt}")
            
            new_entries = self.load_new_entries(last_dt, date_field=date_field)
            logger.info(f"Found {len(new_entries)} new records for {self.symbol}")
            
            if not new_entries:
                return False
                
            # Upload the new slice
            batch = self.upload_slice(vector_store_id, new_entries)
            
            if batch:
                logger.info(f"Upload status for {self.symbol}: {batch.status}")
                
                # Get the latest date - use the date field if it exists
                # or use today's date if we added it ourselves
                if any(date_field in rec for rec in new_entries):
                    latest_date = max(datetime.fromisoformat(rec[date_field]) for rec in new_entries)
                else:
                    latest_date = datetime.now().replace(microsecond=0)
                    
                self.set_last_uploaded_date(latest_date)
                logger.info(f"Checkpoint for {self.symbol} bumped to {latest_date.date()}")
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error processing {self.symbol}: {str(e)}")
            return False


def load_symbol_list():
    """Load the list of symbols from the database files."""
    symbols = []
    db_configs = [
        ("../stocks.db", "SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%'"),
        ("../etf.db", "SELECT DISTINCT symbol FROM etfs"),
        ("../index.db", "SELECT DISTINCT symbol FROM indices")
    ]
    
    for db_file, query in db_configs:
        try:
            con = sqlite3.connect(db_file)
            cur = con.cursor()
            cur.execute(query)
            symbols.extend([r[0] for r in cur.fetchall()])
            con.close()
        except Exception as e:
            logger.error(f"Error connecting to {db_file}: {e}")
            continue
            
    return symbols


def process_symbol(symbol, data_type="historical-price", timeframe="max", date_field="time"):
    """Process a single symbol (for use with concurrent execution)."""
    try:
        uploader = VectorStoreUploader(
            symbol=symbol,
            data_type=data_type,
            timeframe=timeframe
        )
        result = uploader.process(date_field=date_field)
        return (symbol, result)
    except Exception as e:
        logger.error(f"Error processing {data_type} for {symbol}: {e}")
        return (symbol, False)


def process_historical_prices_concurrent(symbols=None, timeframe="max", date_field="time", max_workers=MAX_WORKERS):
    """Process historical price data for specified symbols concurrently."""
    if symbols is None:
        symbols = load_symbol_list()
    
    logger.info(f"Processing {len(symbols)} symbols for historical price data using {max_workers} workers")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_symbol = {
            executor.submit(
                process_symbol, 
                symbol, 
                "historical-price", 
                timeframe, 
                date_field
            ): symbol for symbol in symbols
        }
        
        # Process completed tasks as they finish
        for i, future in enumerate(concurrent.futures.as_completed(future_to_symbol)):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed historical price {i+1}/{len(symbols)}: {symbol}")
            except Exception as e:
                logger.error(f"Task for historical price {symbol} generated an exception: {e}")
    
    success_count = sum(1 for _, success in results if success)
    logger.info(f"Successfully processed {success_count} out of {len(symbols)} symbols for historical prices")
    return results


def process_similar_stocks_concurrent(symbols=None, date_field="time", max_workers=MAX_WORKERS):
    """Process similar-stocks data for specified symbols concurrently."""
    if symbols is None:
        symbols = load_symbol_list()
    
    logger.info(f"Processing {len(symbols)} symbols for similar-stocks data using {max_workers} workers")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_symbol = {
            executor.submit(
                process_symbol, 
                symbol, 
                "similar-stocks", 
                "max",  # timeframe doesn't matter for similar-stocks
                date_field
            ): symbol for symbol in symbols
        }
        
        # Process completed tasks as they finish
        for i, future in enumerate(concurrent.futures.as_completed(future_to_symbol)):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed similar-stocks {i+1}/{len(symbols)}: {symbol}")
            except Exception as e:
                logger.error(f"Task for similar-stocks {symbol} generated an exception: {e}")
    
    success_count = sum(1 for _, success in results if success)
    logger.info(f"Successfully processed {success_count} out of {len(symbols)} symbols for similar stocks")
    return results


if __name__ == "__main__":
    # Configure what you want to run here
    
    # Example usage:
    #specific_symbols = ["AAPL", "MSFT", "GME", "NVDA", "TSLA", "AMZN", "GOOG", "META"]
    
    # Process historical prices concurrently
    #process_historical_prices_concurrent(max_workers=5)
    
    # Process similar stocks concurrently
    process_similar_stocks_concurrent(max_workers=5)