from dotenv import load_dotenv
import os
import subprocess

load_dotenv()

useast_ip_address = os.getenv('USEAST_IP_ADDRESS')


command = [
    "sudo", "rsync", "-avz", "-e", "ssh",
    "/root/backend/app/json/similar-stocks",
    f"root@{useast_ip_address}:/root/backend/app/json"
]
subprocess.run(command)