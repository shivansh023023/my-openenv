import requests
import os
import subprocess
import time

BASE_URL = "http://localhost:8000"

def log(msg, success=True):
    color = "\033[92m" if success else "\033[91m"
    print(f"{color}[{'PASS' if success else 'FAIL'}] {msg}\033[0m")

def run_checks():
    print("=== Hackathon Pre-submission Validation ===")
    
    # 1. ping
    try:
        res = requests.get(f"{BASE_URL}/")
        if res.status_code == 200:
            log("Automated ping (/): Returned 200 OK")
        else:
            log(f"Automated ping (/): {res.status_code}", False)
    except Exception as e:
        log(f"Automated ping failed: {e}", False)

    # 2. Spec Compliance
    if os.path.exists("openenv.yaml"):
        log("openenv.yaml exists")
    else:
        log("openenv.yaml missing", False)
        
    try:
        res = requests.get(f"{BASE_URL}/state")
        res.raise_for_status()
        res = requests.post(f"{BASE_URL}/reset", json={"task": "easy"})
        res.raise_for_status()
        log("State and Reset API Endpoints are functional")
    except Exception as e:
        log("State/Reset validation failed", False)

    # 3. Dockerfile builds
    if os.path.exists("Dockerfile"):
        log("Dockerfile exists")
        print("Testing Docker Build (this may take a moment)...")
        r = subprocess.run(["docker", "build", "-t", "openenv-test-build", "."], capture_output=True, text=True)
        if r.returncode == 0:
            log("Dockerfile successfully built image!")
        else:
            log(f"Dockerfile build failed:\n{r.stderr}", False)
    else:
        log("Dockerfile missing", False)

    # 4. inference.py reproduces without error
    if os.path.exists("inference.py"):
        log("inference.py exists in root folder")
        print("Running inference.py inside the environment for reproduction...")
        # Since server is already running inside docker openenv-server-test with port 8000 mapped
        # We can just run it locally right here
        # It must complete without error
        r = subprocess.run(["docker", "exec", "openenv-server-test", "python3", "inference.py"], capture_output=True, text=True)
        if r.returncode == 0 and "Reward Score:" in r.stdout:
            log("Inference script completed without terminal crash and produced scores")
            print(r.stdout.strip().split("\n")[-2:])
        else:
            log(f"Inference script crashed or didn't produce score: {r.stderr}", False)
    else:
        log("inference.py missing! Did you rename it?", False)

    # 5. Tasks and graders
    try:
        res = requests.get(f"{BASE_URL}/tasks")
        tasks = res.json().get("tasks", {})
        if len(tasks) >= 3:
            log(f"Found {len(tasks)} tasks enumerated in the environment (Required >= 3)")
        else:
            log(f"Found only {len(tasks)} tasks", False)
            
        res = requests.get(f"{BASE_URL}/grader")
        score = res.json().get("score", -1)
        if 0.0 <= score <= 1.0:
            log(f"Grader endpoint returns valid float range score ({score})")
        else:
            log(f"Grader endpoint returns invalid score {score}", False)
    except Exception as e:
        log(f"Task grader validation failed: {e}", False)

if __name__ == "__main__":
    run_checks()
