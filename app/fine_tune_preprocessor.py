from pocketbase import PocketBase  # Client also works the same
import asyncio
import aiohttp
import ujson
import os
from dotenv import load_dotenv

load_dotenv()

async def save_json(name, data):
    path = f"json/fine-tune-model/{name}.jsonl"
    directory = os.path.dirname(path)

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Write each data entry as a separate line in the JSONL file
    with open(path, 'w') as file:
        for entry in data:
            ujson.dump(entry, file)
            file.write('\n')  # Write each JSON object as a new line


pb_admin_email = os.getenv('POCKETBASE_ADMIN_EMAIL')
pb_password = os.getenv('POCKETBASE_PASSWORD')

pb = PocketBase('http://127.0.0.1:8090')
admin_data = pb.collection('_superusers').auth_with_password(pb_admin_email, pb_password)

async def run():
    result = pb.collection('articles').get_full_list(query_params={"filter": f"category='term'"})

    jsonl_data = []  # Initialize an empty list to hold the data

    for item in result:
        # Assuming item has 'prompt' and 'description' attributes
        prompt = item.prompt
        description = item.description
   
        # Create the structured data for each entry
        jsonl_dict = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": description}
            ]
        }
        
        # Append the entry to the list
        jsonl_data.append(jsonl_dict)

    if jsonl_data:
        await save_json('file-financial-term', jsonl_data)

try:
    asyncio.run(run())
except Exception as e:
    print(e)
