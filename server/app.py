from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env.environment import TrustHireEnv

app = FastAPI(title="TrustHireEnv")
env = TrustHireEnv(difficulty="easy", seed=42)


class StepRequest(BaseModel):
    flag_level: str
    next_step: str
    rationale: Optional[str] = ""


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/reset")
def reset(payload: dict | None = None):
    global env
    difficulty = "easy"
    seed = 42

    if payload:
        difficulty = payload.get("difficulty", "easy")
        seed = payload.get("seed", 42)

    env = TrustHireEnv(difficulty=difficulty, seed=seed)
    return {"observation": env.reset()}


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


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)