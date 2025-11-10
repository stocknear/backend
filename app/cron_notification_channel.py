from tqdm import tqdm
import asyncio
from pocketbase import PocketBase  # Client also works the same

from dotenv import load_dotenv
import os

load_dotenv()

pb_admin_email = os.getenv('POCKETBASE_ADMIN_EMAIL')
pb_password = os.getenv('POCKETBASE_PASSWORD')


pb = PocketBase('http://127.0.0.1:8090')
admin_data = pb.collection('_superusers').auth_with_password(pb_admin_email, pb_password)


#manually fix and fill the data for notificationChannels
#default setting is subscribed to all channels

async def subscribe(user_id):
    
    try:        
        result = pb.collection("notificationChannels").get_full_list(query_params={"filter": f"user='{user_id}'"})
        exist = any(item.user == user_id for item in result)

        if exist == False:
            pb.collection("notificationChannels").create({
                'user': user_id,
                'earningsSurprise': True,
                'wiim': True,
                })

    except Exception as e:
        print(e)



async def run():
    all_users = pb.collection("users").get_full_list()
    for item in tqdm(all_users):
        user_id = item.id
        await subscribe(user_id=user_id)       

try:
    asyncio.run(run())
except Exception as e:
    print(e)