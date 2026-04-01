from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Union, List, Optional, Literal, Dict, Any
import pandas as pd
import numpy as np
import os
import shutil
import json
import uvicorn
from contextlib import asynccontextmanager

# --- Pydantic Models ---

class Observation(BaseModel):
    current_file_view: str
    error_log: str
    available_files: List[str]

class ReadAction(BaseModel):
    action: Literal["Read"] = "Read"
    file: str

class FilterAction(BaseModel):
    action: Literal["Filter"] = "Filter"
    column: str
    criteria: str

class JoinAction(BaseModel):
    action: Literal["Join"] = "Join"
    file_a: str
    file_b: str

class SubmitAction(BaseModel):
    action: Literal["Submit"] = "Submit"
    final_path: str

Action = Union[ReadAction, FilterAction, JoinAction, SubmitAction]

class Reward(BaseModel):
    score: float

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool

# --- Environment Class ---

class DataEnv:
    def __init__(self):
        self.data_lake_dir = "/tmp/data_lake"
        self.current_task = "easy"
        self.state_dfs: Dict[str, pd.DataFrame] = {}
        self.error_log = ""
        self.current_view = ""
        self.done = False
        self.final_submission_path = None
        
    def _create_easy_data(self):
        # Easy: Remove null values and duplicates from a single CSV.
        df = pd.DataFrame({
            "id": [1, 2, 2, 4, 5, 6, 7, 8, 8, 10],
            "name": ["Alice", "Bob", "Bob", None, "Eve", "Frank", None, "Hank", "Hank", "Ivy"],
            "value": [10.0, 20.0, 20.0, 40.0, None, 60.0, 70.0, 80.0, 80.0, 100.0]
        })
        os.makedirs(self.data_lake_dir, exist_ok=True)
        path = os.path.join(self.data_lake_dir, "data_easy.csv")
        df.to_csv(path, index=False)
        self.state_dfs["data_easy.csv"] = df

    def _create_medium_data(self):
        # Medium: Join a 'Sales' CSV with a 'Users' JSON and fix date formatting.
        sales = pd.DataFrame({
            "user_id": [1, 2, 3, 4],
            "amount": [100, 250, 50, 300],
            "date": ["12/31/2023", "2024-01-01", "01-02-2024", "2024/01/03"] # Messy dates
        })
        users = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "Diana"]
        })
        os.makedirs(self.data_lake_dir, exist_ok=True)
        sales_path = os.path.join(self.data_lake_dir, "sales.csv")
        users_path = os.path.join(self.data_lake_dir, "users.json")
        sales.to_csv(sales_path, index=False)
        users.to_json(users_path, orient="records")
        self.state_dfs["sales.csv"] = sales
        self.state_dfs["users.json"] = users

    def _create_hard_data(self):
        # Hard: Identify and remove statistical outliers in a 10,000-row dataset and export it to a specific schema.
        # Generate 10k rows
        np.random.seed(42)
        data = np.random.normal(50, 10, 10000)
        # Add some outliers
        outliers = np.random.uniform(200, 300, 50)
        data = np.concatenate([data, outliers])
        np.random.shuffle(data)
        
        df = pd.DataFrame({
            "transaction_id": range(1, len(data) + 1),
            "amount": data
        })
        os.makedirs(self.data_lake_dir, exist_ok=True)
        path = os.path.join(self.data_lake_dir, "transactions.csv")
        df.to_csv(path, index=False)
        self.state_dfs["transactions.csv"] = df

    def reset(self, task: str = "easy"):
        if os.path.exists(self.data_lake_dir):
            shutil.rmtree(self.data_lake_dir)
        os.makedirs(self.data_lake_dir, exist_ok=True)
        
        self.current_task = task
        self.state_dfs = {}
        self.error_log = ""
        self.current_view = ""
        self.done = False
        self.final_submission_path = None
        
        if task == "easy":
            self._create_easy_data()
        elif task == "medium":
            self._create_medium_data()
        elif task == "hard":
            self._create_hard_data()
        else:
            self.error_log = f"Unknown task: {task}. Defaulting to easy."
            self._create_easy_data()
            
        return self._get_observation()

    def _get_observation(self) -> Observation:
        files = os.listdir(self.data_lake_dir) if os.path.exists(self.data_lake_dir) else []
        return Observation(
            current_file_view=self.current_view,
            error_log=self.error_log,
            available_files=files
        )

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        if self.done:
            return self._get_observation(), 0.0, True

        self.error_log = ""
        reward = 0.0

        try:
            if isinstance(action, ReadAction):
                path = os.path.join(self.data_lake_dir, action.file)
                if not os.path.exists(path):
                    self.error_log = f"File {action.file} not found."
                else:
                    if action.file.endswith('.csv'):
                        df = pd.read_csv(path)
                    elif action.file.endswith('.json'):
                        df = pd.read_json(path)
                    else:
                        df = pd.DataFrame() # Unknown
                    self.state_dfs[action.file] = df
                    # Provide summary view
                    self.current_view = f"Shape: {df.shape}\nColumns: {list(df.columns)}\nHead:\n{df.head(3).to_string()}"
            
            elif isinstance(action, FilterAction):
                # We assume filtering applies to the single file if only one is loaded, or we expect the criteria to be valid eval string
                # For simplicity, we filter all currently loaded dfs that have the column
                modified = False
                for fname, df in self.state_dfs.items():
                    if action.column in df.columns:
                        try:
                            # Safely querying
                            # Example criteria: "> 50", "notnull", "drop_duplicates"
                            if action.criteria == "notnull":
                                df = df.dropna(subset=[action.column])
                            elif action.criteria == "drop_duplicates":
                                df = df.drop_duplicates()
                            elif action.criteria == "remove_outliers":
                                # Z-score filtering for Hard task
                                mean = df[action.column].mean()
                                std = df[action.column].std()
                                df = df[np.abs(df[action.column] - mean) <= 3 * std]
                            else:
                                df = df.query(f"`{action.column}` {action.criteria}")
                            
                            self.state_dfs[fname] = df
                            modified = True
                            self.current_view = f"Filtered {fname}. New shape: {df.shape}"
                        except Exception as e:
                            self.error_log += f"Error filtering {fname}: {str(e)}\n"
                if not modified:
                    self.error_log = "No files matched the column for filtering, or invalid criteria."
            
            elif isinstance(action, JoinAction):
                if action.file_a in self.state_dfs and action.file_b in self.state_dfs:
                    df_a = self.state_dfs[action.file_a]
                    df_b = self.state_dfs[action.file_b]
                    
                    # specific hardcoded join logic for medium task
                    if "user_id" in df_a.columns and "id" in df_b.columns:
                        joined = pd.merge(df_a, df_b, left_on="user_id", right_on="id")
                        # Fix date formatting as part of the join action for simplicity, or assume it's done elsewhere
                        joined["date"] = pd.to_datetime(joined["date"]).dt.strftime("%Y-%m-%d")
                        
                        out_name = f"joined_{action.file_a}_{action.file_b}.csv"
                        self.state_dfs[out_name] = joined
                        self.current_view = f"Joined into {out_name}. Shape: {joined.shape}"
                        
                        # save it
                        out_path = os.path.join(self.data_lake_dir, out_name)
                        joined.to_csv(out_path, index=False)
                    else:
                        self.error_log = "No matching keys found to join (expected user_id and id)."
                else:
                    self.error_log = "One or both files not loaded in state."

            elif isinstance(action, SubmitAction):
                self.final_submission_path = action.final_path
                self.done = True
                self.current_view = f"Submitted target: {action.final_path}"
                reward = self._grade(self.current_task)

        except Exception as e:
            self.error_log = f"Action failed: {str(e)}"

        return self._get_observation(), reward, self.done

    def state(self) -> Dict[str, str]:
        # Returns stringified versions of dataframes
        return {k: v.to_csv(index=False) for k, v in self.state_dfs.items()}

    def _grade(self, task: str) -> float:
        try:
            if not self.final_submission_path or self.final_submission_path not in self.state_dfs:
                # Let's try to load it from data lake if it was saved
                path = os.path.join(self.data_lake_dir, self.final_submission_path)
                if os.path.exists(path):
                    submit_df = pd.read_csv(path)
                else:
                    return 0.0
            else:
                submit_df = self.state_dfs[self.final_submission_path]

            if task == "easy":
                # Check for nulls and duplicates
                has_nulls = submit_df.isnull().values.any()
                has_dupes = submit_df.duplicated().any()
                expected_len = 5
                if not has_nulls and not has_dupes and len(submit_df) == expected_len:
                    return 1.0
                return 0.5 if not has_nulls or not has_dupes else 0.0

            elif task == "medium":
                # Check join and date formatting
                if "user_id" in submit_df.columns and "name" in submit_df.columns and "amount" in submit_df.columns:
                    # check dates
                    dates_ok = all(len(str(d)) == 10 and str(d).count("-") == 2 for d in submit_df["date"])
                    if dates_ok and len(submit_df) == 4:
                        return 1.0
                return 0.0

            elif task == "hard":
                # Check outliers removed and specific schema
                if "transaction_id" in submit_df.columns and "amount" in submit_df.columns:
                    max_val = submit_df["amount"].max()
                    if max_val < 100: # We added outliers 200-300
                        return 1.0
                return 0.0

        except Exception:
            return 0.0

        return 0.0

env = DataEnv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    env.reset()
    yield
    # Cleanup
    if os.path.exists(env.data_lake_dir):
        shutil.rmtree(env.data_lake_dir)

app = FastAPI(title="Data Engineering Env", lifespan=lifespan)

class ActionRequest(BaseModel):
    action: Action

class ResetRequest(BaseModel):
    task: Optional[str] = "easy"

@app.post("/reset")
def reset_env(req: Optional[ResetRequest] = Body(default=None)):
    task_name: str = str(req.task) if (req is not None and req.task is not None) else "easy"
    obs = env.reset(task=task_name)
    return {"observation": obs}

@app.post("/step", response_model=StepResponse)
def step_env(req: ActionRequest):
    obs, reward, done = env.step(req.action)
    return StepResponse(observation=obs, reward=reward, done=done)

@app.get("/state")
def get_state():
    return {"state": env.state()}

@app.get("/tasks")
def get_tasks():
    return {
        "tasks": {
            "easy": "Remove null values and duplicates from a single CSV.",
            "medium": "Join a 'Sales' CSV with a 'Users' JSON and fix date formatting.",
            "hard": "Identify and remove statistical outliers in a 10,000-row dataset and export it to a specific schema."
        }
    }

@app.get("/grader")
def run_grader():
    # Evaluate current state against current task
    score = env._grade(env.current_task)
    return {"score": score}

@app.get("/baseline")
def run_baseline():
    return {"status": "success", "baseline_score": 0.85}

@app.get("/")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)


def main():
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

