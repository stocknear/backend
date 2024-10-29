import subprocess

# Function to run commands and log output
def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

def run_tracker():
    # Run Python scripts
    scripts = [
        "cron_reddit_tracker.py",
        "cron_reddit_statistics.py",
        "cron_cramer_tracker.py",
        "cron_lobbying_tracker.py",
        "cron_sentiment_tracker.py"
        "cron_insider_tracker.py"
        "cron_cap_category.py"
    ]
    for script in scripts:
        run_command(["python3", script])


run_tracker()