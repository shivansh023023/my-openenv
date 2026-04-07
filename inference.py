import os
import json
import requests
from openai import OpenAI

# Ensure we read OPENAI_API_KEY from the environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy_key")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_BASE_URL = os.environ.get("API_BASE_URL") # Handled elegantly if None

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=API_BASE_URL,
    timeout=20.0
)

# Interact with the live Hugging Face Space Environment
BASE_URL = "https://hello12334-openenv-data-cleaner.hf.space"

def llm_choose_action(observation, task):
    task_desc = {
        "easy": "Easy (Remove null values and duplicates from a single CSV).",
        "medium": "Medium (Join a 'Sales' CSV with a 'Users' JSON and fix date formatting).",
        "hard": "Hard (Identify and remove statistical outliers in a 10,000-row dataset and export it to a specific schema)."
    }.get(task, task)
    
    prompt = f"""
You are an autonomous AI data engineer. Your current task is: {task_desc}
Observation:
{observation}
You must return a valid JSON object matching the `ActionRequest` schema.
Example Actions:
{{ "action": {{ "action": "Read", "file": "data_easy.csv" }} }}
Or {{ "action": {{ "action": "Filter", "column": "name", "criteria": "notnull" }} }}
Or {{ "action": {{ "action": "Join", "file_a": "sales.csv", "file_b": "users.json" }} }}
Or {{ "action": {{ "action": "Submit", "final_path": "data_easy.csv" }} }}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM API Exception: {e}", flush=True)
        return None

def run_inference():
    print("--- Starting Inference Workflow ---", flush=True)
    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        # 1. At the beginning of each task
        print(f"[START] task={task}", flush=True)
        
        try:
            res = requests.post(f"{BASE_URL}/reset", json={"task": task})
            res.raise_for_status()
            obs = res.json()["observation"]
        except Exception as e:
            print(f"Error resetting env for task {task}: {e}", flush=True)
            print(f"[END] task={task} score=0.0 steps=0", flush=True)
            continue

        done = False
        steps = 0
        reward = 0.0
        
        while not done and steps < 15:
            action = llm_choose_action(obs, task)
            
            # Simple fallback if model output fails
            if action is None:
                action = {"action": {"action": "Submit", "final_path": "fallback_submit.csv"}}
                
            steps += 1
            try:
                res = requests.post(f"{BASE_URL}/step", json=action)
                res.raise_for_status()
                data = res.json()
                obs = data.get("observation", "")
                reward = float(data.get("reward", 0.0))
                done = data.get("done", True)
            except Exception as e:
                print(f"Error executing step: {e}", flush=True)
                done = True
            
            # 2. After every step the agent takes
            print(f"[STEP] step={steps} reward={reward}", flush=True)
            
        # Get final score from the grader endpoint
        try:
            grader_res = requests.get(f"{BASE_URL}/grader")
            grader_res.raise_for_status()
            score = float(grader_res.json().get("score", 0.0))
        except Exception as e:
            print(f"Error getting grader score: {e}", flush=True)
            score = 0.0
            
        # 3. When the task is finished
        print(f"[END] task={task} score={score} steps={steps}", flush=True)

if __name__ == "__main__":
    run_inference()
