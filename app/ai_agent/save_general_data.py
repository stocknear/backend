import os
import json
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import logging
import orjson

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_BASE_DIR = "../"
JSON_SUBDIR = "json"
MAX_WORKERS = 1

# Get vector store ID and OpenAI client
vector_store_id = os.getenv("VECTOR_STORE_ID")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_TYPE = [
    {"name": 'Wallstreet Analyst', "path": "analyst/analyst-db"},
    {"name": 'Congressional Trading', "path": "congress-trading/politician-db"},
    {"name": 'Dividend Calendar', "path": "dividends-calendar"},
    {"name": "Earnings Calendar", "path": "earnings-calendar"},
    {"name": "Economic Calendar", "path": "economic-calendar"},
    {"name": "IPO Calendar", "path": "ipo-calendar"},
    {"name": "POTUS Executive Orders", "path": "executive-orders"},
]

def find_json_files_in_subfolders(base_directory: Path, folder_paths: list[str]) -> list[Path]:
    json_files = []
    if not base_directory.is_dir():
        logger.error(f"Base directory not found: {base_directory}")
        return json_files

    for folder_path in folder_paths:
        subfolder_path = base_directory / folder_path
        if not subfolder_path.is_dir():
            logger.warning(f"Skipping missing: {subfolder_path}")
            continue
        for item in subfolder_path.glob("*.json"):
            if item.is_file():
                json_files.append(item)
    return json_files


def delete_existing_files_in_vector_store(vector_store_id: str, client: OpenAI, max_workers: int = 5):
    if not vector_store_id:
        logger.error("VECTOR_STORE_ID is not set.")
        return
    try:
        iterator = client.vector_stores.files.list(vector_store_id=vector_store_id, limit=100)
        all_files = []
        while True:
            all_files.extend(iterator.data)
            if not iterator.has_more:
                break
            iterator = client.vector_stores.files.list(
                vector_store_id=vector_store_id,
                limit=100,
                after=iterator.last_id
            )
        if not all_files:
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(
                client.vector_stores.files.delete,
                vector_store_id=vector_store_id,
                file_id=f.id
            ): f.id for f in all_files}
            for future in concurrent.futures.as_completed(futures):
                fid = futures[future]
                try:
                    future.result()
                    logger.info(f"Deleted file {fid}")
                except Exception as e:
                    logger.error(f"Failed deleting {fid}: {e}")
    except Exception as e:
        logger.error(f"Error cleaning vector store: {e}")


def upload_single_file(file_path: Path, vector_store_id: str, client: OpenAI, data_types: list[dict]) -> bool:
    if not vector_store_id:
        logger.error("VECTOR_STORE_ID is not set.")
        return False
    if not file_path.exists():
        logger.warning(f"File missing: {file_path}")
        return False

    try:
        # Determine category
        rel = file_path.relative_to(Path(DEFAULT_BASE_DIR) / JSON_SUBDIR)
        folder = str(rel.parent)
        category = next((dt['name'] for dt in data_types if folder.startswith(dt['path'])), 'Unknown')

        # Load original JSON
        orig_data = orjson.loads(file_path.read_bytes())
        # Build full metadata payload
        payload = {
            "source": folder,
            "filename": file_path.name,
            "Category": category,
            "data": orig_data
        }
        # Prepare file tuple: (filename, content_bytes, mime_type)
        file_content = orjson.dumps(payload)
        files = [(file_path.name, file_content, 'application/json')]

        # Upload & poll
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=files,
        )
        logger.info(f"Upload status for {file_path}: {batch.status}")
        return batch.status == 'completed'
    except Exception as e:
        logger.error(f"Upload error {file_path}: {e}")
        return False


def process_selected_json_subfolders_concurrent(data_types: list[dict], base_dir: str = DEFAULT_BASE_DIR, max_workers: int = MAX_WORKERS):
    base = Path(base_dir) / JSON_SUBDIR
    paths = [dt['path'] for dt in data_types]
    files = find_json_files_in_subfolders(base, paths)
    if not files:
        logger.info("No JSON files found.")
        return
    delete_existing_files_in_vector_store(vector_store_id, client, max_workers)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futs = {executor.submit(upload_single_file, f, vector_store_id, client, data_types): f for f in files}
        for fut in concurrent.futures.as_completed(futs):
            fpath = futs[fut]
            success = fut.result()
            results.append((fpath, success))
            logger.info(f"{fpath} - {'OK' if success else 'FAILED'}")
    count = sum(1 for _, s in results if s)
    logger.info(f"Uploaded {count}/{len(files)} files.")


if __name__ == "__main__":
    process_selected_json_subfolders_concurrent(DATA_TYPE)
