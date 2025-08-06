import os
import subprocess
import time
from datetime import datetime

# Your local folder where CSV files are stored
REPO_PATH = r"C:\PLCData"

def run_command(command, cwd=None):
    """Run shell commands in the specified directory."""
    result = subprocess.run(command, cwd=cwd, text=True, shell=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error running '{command}':\n{result.stderr}")
    return result.stdout.strip()

def stage_csv_files():
    """Add only CSV files to git staging."""
    for root, dirs, files in os.walk(REPO_PATH):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.relpath(os.path.join(root, file), REPO_PATH)
                run_command(f'git add "{full_path}"', cwd=REPO_PATH)

def push_csv_to_github():
    os.chdir(REPO_PATH)
    stage_csv_files()

    status = run_command("git diff --cached --name-only", cwd=REPO_PATH)

    if status:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run_command(f'git commit -m "CSV update: {timestamp}"', cwd=REPO_PATH)
        run_command("git push", cwd=REPO_PATH)
        print(f"[{timestamp}] Pushed .csv files to GitHub.")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] No new CSV files to push.")

if __name__ == "__main__":
    while True:
        push_csv_to_github()
        time.sleep(3)
