import os
import json
import concurrent.futures
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

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_BASE_DIR = "../"
JSON_SUBDIR = "json" # Subdirectory within base_dir where the target subfolders reside
MAX_WORKERS = 5 # Adjust based on your API rate limits and system resources

# Get vector store ID and OpenAI client from environment variables
vector_store_id = os.getenv("VECTOR_STORE_ID")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def find_json_files_in_subfolders(base_directory: Path, subfolder_names: list[str]) -> list[Path]:
    """
    Finds all .json files within specified subfolders of the base directory.
    Does NOT search recursively within these subfolders.

    Args:
        base_directory: The base directory (e.g., Path("../json")).
        subfolder_names: A list of names of the subfolders to search within.

    Returns:
        A list of Path objects for all found .json files in the specified subfolders.
    """
    json_files = []
    if not base_directory.is_dir():
        logger.error(f"Base directory not found: {base_directory}")
        return json_files

    logger.info(f"Searching for .json files in specified subfolders under: {base_directory}")

    for subfolder_name in subfolder_names:
        subfolder_path = base_directory / subfolder_name
        if not subfolder_path.is_dir():
            logger.warning(f"Specified subfolder not found, skipping: {subfolder_path}")
            continue

        logger.info(f"Searching in subfolder: {subfolder_path}")
        # Use glob instead of rglob to avoid recursive search within the subfolder
        for item in subfolder_path.glob("*.json"):
            if item.is_file():
                json_files.append(item)
                # logger.debug(f"Found file: {item}") # Uncomment for detailed file listing

    logger.info(f"Found {len(json_files)} .json files in the specified subfolders.")
    return json_files

def delete_existing_files_in_vector_store(vector_store_id: str, client: OpenAI, max_workers: int = 5):
    """
    Deletes all existing files in the specified vector store concurrently.
    Args:
        vector_store_id: The ID of the vector store.
        client: The OpenAI client instance.
        max_workers: Number of threads for concurrent deletions.
    """
    if not vector_store_id:
        logger.error("Vector store ID is not set. Cannot delete existing files.")
        return

    logger.info(f"Attempting to delete all existing files in vector store: {vector_store_id}")
    try:
        # List all files
        files_iterator = client.vector_stores.files.list(vector_store_id=vector_store_id, limit=100)
        all_files = []

        while True:
            all_files.extend(files_iterator.data)
            if not files_iterator.has_more:
                break
            files_iterator = client.vector_stores.files.list(
                vector_store_id=vector_store_id,
                limit=100,
                after=files_iterator.last_id
            )

        if not all_files:
            logger.info("No existing files found in the vector store.")
            return

        logger.info(f"Found {len(all_files)} existing files to delete. Deleting in parallel...")

        # Use concurrent deletion
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    client.vector_stores.files.delete,
                    vector_store_id=vector_store_id,
                    file_id=file.id
                ): file.id for file in all_files
            }

            for future in concurrent.futures.as_completed(futures):
                file_id = futures[future]
                try:
                    future.result()
                    logger.info(f"Deleted file {file_id}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_id}: {e}")

        logger.info("Finished deleting files from vector store.")

    except Exception as e:
        logger.error(f"Error listing or deleting files from vector store {vector_store_id}: {e}")


def upload_single_file(file_path: Path, vector_store_id: str, client: OpenAI) -> bool:
    """
    Uploads a single file to the specified vector store.

    Args:
        file_path: The path to the file to upload.
        vector_store_id: The ID of the vector store.
        client: The OpenAI client instance.

    Returns:
        True if the upload was successful, False otherwise.
    """
    if not vector_store_id:
        logger.error(f"Vector store ID is not set. Cannot upload file: {file_path}")
        return False

    if not file_path.exists():
        logger.warning(f"File not found at {file_path}")
        return False

    try:
        logger.info(f"Uploading file: {file_path}")
        with open(file_path, "rb") as f:
            # Use upload_and_poll to wait for processing to complete
            batch = client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id,
                files=[f],
            )
        logger.info(f"Upload status for {file_path}: {batch.status}")
        # Check batch status for success indicators like 'completed'
        if batch.status == 'completed':
             return True
        else:
             logger.error(f"File upload batch failed or did not complete for {file_path}. Status: {batch.status}")
             # You might want to inspect batch.file_counts for details on failed files
             return False

    except Exception as e:
        logger.error(f"Error uploading file {file_path} to vector store {vector_store_id}: {e}")
        return False

def process_selected_json_subfolders_concurrent(subfolder_names: list[str], base_dir: str = DEFAULT_BASE_DIR, max_workers: int = MAX_WORKERS):
    json_base_directory = Path(base_dir) / JSON_SUBDIR

    # Find all JSON files in the specified subfolders
    json_files_to_upload = find_json_files_in_subfolders(json_base_directory, subfolder_names)

    if not json_files_to_upload:
        logger.info(f"No JSON files found in the specified subfolders: {subfolder_names}. Exiting.")
        return

    delete_existing_files_in_vector_store(vector_store_id, client, MAX_WORKERS)

    logger.info(f"Starting concurrent upload of {len(json_files_to_upload)} files from specified subfolders using {max_workers} workers.")

    results = []
    # Use ThreadPoolExecutor for concurrent file uploads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit upload tasks for each file
        future_to_file = {
            executor.submit(upload_single_file, file_path, vector_store_id, client): file_path
            for file_path in json_files_to_upload
        }

        # Process results as tasks complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                success = future.result()
                results.append((file_path, success))
                status = "SUCCESS" if success else "FAILED"
                logger.info(f"Completed processing {i+1}/{len(json_files_to_upload)}: {file_path} - {status}")
            except Exception as e:
                logger.error(f"Task for file {file_path} generated an exception: {e}")
                results.append((file_path, False)) # Mark as failed if exception occurs

    success_count = sum(1 for _, success in results if success)
    logger.info(f"Finished processing. Successfully uploaded {success_count} out of {len(json_files_to_upload)} files from specified subfolders.")


if __name__ == "__main__":
    # Example usage: Specify the subfolders you want to process
    # Replace with the actual names of the subfolders you want to include
    subfolders_to_process = [
        "analyst/analyst-db",
        "congress-trading/politician-db"
    ]

    # Process JSON files only from the specified subfolders
    process_selected_json_subfolders_concurrent(
        subfolder_names=subfolders_to_process,
        base_dir=DEFAULT_BASE_DIR,
        max_workers=MAX_WORKERS
    )