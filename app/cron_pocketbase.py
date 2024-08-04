from datetime import datetime, timedelta
from pocketbase import PocketBase  # Client also works the same
import asyncio
from dotenv import load_dotenv
import os
load_dotenv()

pb_admin_email = os.getenv('POCKETBASE_ADMIN_EMAIL')
pb_password = os.getenv('POCKETBASE_PASSWORD')

pb = PocketBase('http://127.0.0.1:8090')
admin_data = pb.admins.auth_with_password(pb_admin_email, pb_password)


now = datetime.now()
one_month_ago = now - timedelta(days=30)


async def update_free_trial():

    data =  pb.collection("users").get_full_list(query_params = {"filter": f'freeTrial = True'})

    for item in data:
        created_date = item.created
        # Check if the created date is more than N days ago
        if created_date < one_month_ago:
            # Update the user record
            pb.collection("users").update(item.id, {
                "tier": 'Free',
                "freeTrial": False,
            })

asyncio.run(update_free_trial())