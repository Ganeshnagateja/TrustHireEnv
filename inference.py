import os
import json
from baseline_eval import main as baseline_main

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

def run():
    """
    Meta Hackathon automated inference entrypoint.
    Uses deterministic no-LLM baseline for reproducibility.
    """
    os.system("python baseline_eval.py --no-llm")

if __name__ == "__main__":
    run()