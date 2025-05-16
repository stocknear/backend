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
    def __init__(self, symbol, base_dir=DEFAULT_BASE_DIR):
        """
        Initialize the uploader with configuration for a specific symbol.
        
        Args:
            symbol: The stock/ETF/index symbol (e.g., 'GME')
            base_dir: Base directory for all data
        """
        self.symbol = symbol
        self.base_dir = Path(base_dir)
        
        # Set up OpenAI client
        self.vector_store_id = os.getenv("VECTOR_STORE_ID")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
        # Define file paths for different data types
        self.historical_price_path = self.base_dir / f"json/historical-price/max/{symbol}.json"
        self.similar_stocks_path = self.base_dir / f"json/similar-stocks/{symbol}.json"
        
        # Path for combined data file (temporary)
        self.combined_data_path = self.base_dir / f"json/combined/{symbol}.json"
        
        # Ensure the combined directory exists
        os.makedirs(self.base_dir / "json/combined", exist_ok=True)
    
    def load_data(self, file_path):
        """Load data from a file if it exists."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading data from {file_path}: {e}")
        return None
        
    def combine_data(self):
        """Combine historical price and similar stocks data into a single structure."""
        # Load historical price data
        historical_data = self.load_data(self.historical_price_path)
        
        # Load similar stocks data
        similar_stocks_data = self.load_data(self.similar_stocks_path)
        
        # Create combined data structure
        combined_data = {
            "Symbol": self.symbol,
            "historical-price": historical_data or [],
            "similar-stocks": similar_stocks_data or {}
        }
        
        # Save combined data to file
        try:
            with open(self.combined_data_path, 'w') as f:
                json.dump(combined_data, f, indent=2)
            logger.info(f"Combined data for {self.symbol} saved to {self.combined_data_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving combined data for {self.symbol}: {e}")
            return False

            
    def delete_existing_files(self, vector_store_id: str):
        """Delete all existing files in the vector store."""
        try:
            files = self.client.vector_stores.files.list(vector_store_id=vector_store_id)
       
            for file in files.data:
                self.client.vector_stores.files.delete(vector_store_id=vector_store_id, file_id=file.id)
                logger.info(f"Deleted file {file.id} from vector store {vector_store_id}")
        except Exception as e:
            logger.error(f"Error deleting files from vector store {vector_store_id}: {e}")

    def upload_file(self, vector_store_id: str):
        """Upload the combined data file to the vector store."""
        if not os.path.exists(self.combined_data_path):
            logger.warning(f"Combined data file not found at {self.combined_data_path}")
            return None

        try:
            with open(self.combined_data_path, "rb") as f:
                batch = self.client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store_id,
                    files=[f],
                )
            return batch
        except Exception as e:
            logger.error(f"Error uploading file to vector store {vector_store_id}: {e}")
            return None

    def process(self):
        """Process a single symbol's data upload."""
        try:
            # First combine the data from different sources
            if not self.combine_data():
                logger.error(f"Failed to combine data for {self.symbol}")
                return False
                
            # Get or create vector store ID
            vector_store_id = self.vector_store_id
            logger.info(f"Processing combined data for {self.symbol}")

            # Delete existing files
            self.delete_existing_files(vector_store_id)

            # Upload the new combined file
            batch = self.upload_file(vector_store_id)

            if batch:
                logger.info(f"Upload status for {self.symbol}: {batch.status}")
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

def process_symbol(symbol):
    """Process a single symbol (for use with concurrent execution)."""
    try:
        uploader = VectorStoreUploader(symbol=symbol)
        result = uploader.process()
        return (symbol, result)
    except Exception as e:
        logger.error(f"Error processing data for {symbol}: {e}")
        return (symbol, False)

def process_symbols_concurrent(symbols=None, max_workers=MAX_WORKERS):
    """Process all data types for specified symbols concurrently."""
    if symbols is None:
        symbols = load_symbol_list()
    
    logger.info(f"Processing {len(symbols)} symbols using {max_workers} workers")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_symbol = {
            executor.submit(process_symbol, symbol): symbol for symbol in symbols
        }
        
        # Process completed tasks as they finish
        for i, future in enumerate(concurrent.futures.as_completed(future_to_symbol)):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed processing {i+1}/{len(symbols)}: {symbol}")
            except Exception as e:
                logger.error(f"Task for {symbol} generated an exception: {e}")
    
    success_count = sum(1 for _, success in results if success)
    logger.info(f"Successfully processed {success_count} out of {len(symbols)} symbols")
    return results

if __name__ == "__main__":
    # Example usage:
    symbols = ["AAPL",'AMD','TSLA']

    # Process all data types concurrently
    process_symbols_concurrent(symbols=symbols, max_workers=5)
