from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
import os

from env.environment import TrustHireEnv

# Required env vars pattern for checker
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "TrustHireEnv")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional

app = FastAPI(title="TrustHireEnv OpenEnv API")

env = TrustHireEnv(difficulty="easy", seed=42)


class StepRequest(BaseModel):
    flag_level: str
    next_step: str
    rationale: str | None = ""


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/reset")
def reset(payload: Dict[str, Any] | None = None):
    global env
    difficulty = "easy"
    seed = 42

    if payload:
        difficulty = payload.get("difficulty", "easy")
        seed = payload.get("seed", 42)

    env = TrustHireEnv(difficulty=difficulty, seed=seed)
    obs = env.reset()
    return {"observation": obs}


@app.post("/step")
def step(req: StepRequest):
    obs, reward, done, info = env.step({
        "flag_level": req.flag_level,
        "next_step": req.next_step,
        "rationale": req.rationale,
    })
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/health")
def health_check():
    return {"healthy": True}