import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment
load_dotenv()

vector_store_id = OpenAI(api_key=os.getenv("VECTOR_STORE_ID"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VECTOR_STORE_NAME = "GME Historical Prices"
HIST_JSON_PATH = "../json/historical-price/max/GME.json"
STATE_PATH = "../json/vector-store/historical-price/max/GME.json"

def get_last_uploaded_date():
    """Read the last-uploaded ISO date from STATE_PATH, or return epoch if missing."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return datetime.fromisoformat(f.read().strip())
    else:
        # If first run, use a very old date so you upload everything
        return datetime.fromisoformat("1970-01-01T00:00:00")

def set_last_uploaded_date(dt: datetime):
    """Write the new last-uploaded ISO date to STATE_PATH, creating directories if needed."""
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        f.write(dt.isoformat())

def load_new_entries(last_dt: datetime):
    """Load the JSON file and return only entries newer than last_dt."""
    with open(HIST_JSON_PATH, "r") as f:
        data = json.load(f)
    # Assuming each record has a "date" field in "YYYY-MM-DD" format:
    new_records = [
        rec for rec in data
        if datetime.fromisoformat(rec["time"]) > last_dt
    ]
    return new_records

def get_or_create_store_id(name: str):
    resp = client.vector_stores.list()
    for vs in resp.data:
        if vs.name == name:
            print("Found Existing Vector Store...")
            return vs.id

    print("No Vector Store available. Creating a new one...")
    new_vs = client.vector_stores.create(name=name)
    return new_vs.id

def upload_slice(vector_store_id: str, slice_records: list):
    """Upload the filtered slice to the vector store as one batch."""
    if not slice_records:
        print("No new records to upload.")
        return None

    # Write slice to a temp file
    tmp_path = HIST_JSON_PATH + ".slice.json"
    with open(tmp_path, "w") as f:
        json.dump(slice_records, f)

    with open(tmp_path, "rb") as f:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=[f],
        )
    os.remove(tmp_path)
    return batch

if __name__ == "__main__":
    # 1) Find or create the store
    #vector_store_id = get_or_create_store_id(VECTOR_STORE_NAME)
    
    # 2) Load checkpoint & filter JSON
    last_dt = get_last_uploaded_date()
    new_entries = load_new_entries(last_dt)
    
    # 3) Upload only the new slice
    batch = upload_slice(vector_store_id, new_entries)
    if batch:
        print("Upload status:", batch.status)
        # 4) Update the checkpoint to the latest date we just sent
        latest_date = max(datetime.fromisoformat(rec["time"]) for rec in new_entries)
        set_last_uploaded_date(latest_date)
        print(f"Checkpoint bumped to {latest_date.date()}")
