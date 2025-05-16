import os
import json
import sqlite3
import concurrent.futures
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import logging
import shutil
import time


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
MAX_WORKERS = 2 # Adjust based on your API rate limits and system resources

vector_store_id = os.getenv("VECTOR_STORE_ID")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class VectorStoreUploader:
    # map a “logical name” to the sub-path under base_dir/json
    DATA_TYPES = {
        "company_profile": "stockdeck/{symbol}.json",
        "historical_price": "historical-price/max/{symbol}.json",
        "historical_employees": "historical-employees/{symbol}.json",
        "historical_market_cap": "market-cap/companies/{symbol}.json",
        "similar_stocks": "similar-stocks/{symbol}.json",
        "business_metrics": "business-metrics/{symbol}.json",
        "share_statistics": "share-statistics/{symbol}.json",
        "financial_score": "financial-score/{symbol}.json",
        "earnings_next": "earnings/next/{symbol}.json",
        "earnings_past": "earnings/past/{symbol}.json",
        "earnings_surprise": "earnings/surprise/{symbol}.json",
        "dividends": "dividends/companies/{symbol}.json",
        "analyst_estimate": "analyst-estimate/{symbol}.json",
        "income_statement_annual": "financial-statements/income-statement/annual/{symbol}.json",
        "balance_sheet_statement_annual": "financial-statements/balance-sheet-statement/annual/{symbol}.json",
        "cash_flow_statement_annual": "financial-statements/cash-flow-statement/annual/{symbol}.json",
        "ratios_statement_annual": "financial-statements/ratios/annual/{symbol}.json",
        "key_metrics_annual": "financial-statements/key-metrics/annual/{symbol}.json",
        "ownership_stats": "ownership-stats/{symbol}.json",
        "fail_to_deliver": "fail-to-deliver/companies/{symbol}.json",
        "why_priced_moved": "wiim/company/{symbol}.json",
        "historical_option_unusual_activtiy": "unusual-activity/{symbol}.json",
        "option_max_pain": "max-pain/{symbol}.json",
        "option_hottest_contracts": "hottest-contracts/companies/{symbol}.json",
        "option_open_interest_by_expiry": "oi/expiry/{symbol}.json",
        "option_open_interest_by_strike": "oi/strike/{symbol}.json",
        "option_gex_dex_overview": "gex-dex/overview/{symbol}.json",
        "option_gex_by_strike_gex": "gex-dex/strike/gex/{symbol}.json",
        "option_dex_by_strike": "gex-dex/strike/dex/{symbol}.json",
        "option_gex_by_expiry_gex": "gex-dex/expiry/gex/{symbol}.json",
        "option_dex_by_expiry": "gex-dex/expiry/dex/{symbol}.json",
        "historical_option_overview": "options-historical-data/companies/{symbol}.json",
        "shareholders": "shareholders/{symbol}.json",
        "historical_dark_pool_data": "dark-pool/companies/{symbol}.json",
        "historical_enterprise_values": "enterprise-values/{symbol}.json",
        "analyst_ratings": "analyst/history/{symbol}.json",
        "congress_trading": "congress-trading/company/{symbol}.json",
        "insider_trading": "insider-trading/history/{symbol}.json",
    }

    def __init__(self, symbol: str, base_dir: str = DEFAULT_BASE_DIR):
        self.symbol = symbol.upper()
        self.base_dir = Path(base_dir)
        self.vector_store_id = vector_store_id
        self.client = client

        # Build a dict of full paths for each data type
        self.paths = {
            key: self.base_dir / "json" / pattern.format(symbol=self.symbol)
            for key, pattern in self.DATA_TYPES.items()
        }

        # Ensure combined directory exists and set combined_data_path
        combined_dir = self.base_dir / "json" / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        self.combined_data_path = combined_dir / f"{self.symbol}.json"
        self.paths["combined"] = self.combined_data_path

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
        """
        Load all configured data files, merge into a big dict,
        and dump it to combined/{symbol}.json.
        """
        combined = {"symbol": self.symbol}

        for key, path in self.paths.items():
            if key == "combined":
                continue  # skip the output file
            data = self.load_data(path)
            combined[key] = data if data is not None else {}

        # Write out the merged JSON
        out_path = self.paths["combined"]
        try:
            with open(out_path, "w") as f:
                json.dump(combined, f, indent=2)
        except Exception as e:
            logger.error(f"Could not write combined data to {out_path}: {e}")

        #print(combined)
        #time.sleep(1000)
        return combined

            
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
        ("../stocks.db", "SELECT DISTINCT symbol FROM stocks WHERE symbol NOT LIKE '%.%' AND marketCap > 1E9"),
        #("../etf.db", "SELECT DISTINCT symbol FROM etfs"),
        #("../index.db", "SELECT DISTINCT symbol FROM indices")
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
    symbols = ["AAPL",'GME','AMD','TSLA']

    # Process all data types concurrently
    process_symbols_concurrent(symbols=symbols, max_workers=5)
    try:
        shutil.rmtree("../json/combined")
    except:
        pass