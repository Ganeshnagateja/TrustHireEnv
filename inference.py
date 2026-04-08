import os
from typing import List, Dict, Any

from openai import OpenAI
from env.environment import TrustHireEnv

# =========================================================
# REQUIRED ENV VARIABLES
# =========================================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")

TASK_NAME = "trusthire"
BENCHMARK = "TrustHireEnv"
MAX_STEPS = 5


# =========================================================
# MANDATORY STDOUT FORMAT
# =========================================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.3f} rewards={reward_str}",
        flush=True,
    )


# =========================================================
# LLM POLICY
# =========================================================
def choose_action_with_llm(
    client: OpenAI,
    observation: Any,
) -> Dict[str, str]:
    """
    Uses OpenAI client as required by Meta instructions.
    Falls back safely if API is unavailable.
    """
    prompt = f"""
You are solving TrustHireEnv.

Observation:
{observation}

Return exactly one risk decision:
high|escalate|reason
medium|manual_review|reason
low|approve|reason
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an interview fraud detection agent."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=50,
        )

        text = response.choices[0].message.content.strip()

        parts = text.split("|")
        if len(parts) >= 3:
            return {
                "flag_level": parts[0].strip(),
                "next_step": parts[1].strip(),
                "rationale": parts[2].strip(),
            }

    except Exception:
        pass

    # Safe fallback
    obs_text = str(observation).lower()

    if "critical" in obs_text:
        return {
            "flag_level": "high",
            "next_step": "escalate",
            "rationale": "Critical anomaly detected",
        }
    elif "warning" in obs_text:
        return {
            "flag_level": "medium",
            "next_step": "manual_review",
            "rationale": "Suspicious activity detected",
        }
    else:
        return {
            "flag_level": "low",
            "next_step": "approve",
            "rationale": "No major issue found",
        }


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    env = TrustHireEnv(difficulty="easy", seed=42)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        observation = env.reset()

        for step in range(1, MAX_STEPS + 1):
            action = choose_action_with_llm(client, observation)

            observation, reward, done, info = env.step(action)

            reward = float(reward)
            rewards.append(reward)
            steps_taken = step

            action_str = (
                f"{action['flag_level']}|"
                f"{action['next_step']}|"
                f"{action['rationale']}"
            )

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=bool(done),
                error=None,
            )

            if done:
                break

        total_reward = sum(rewards)

        # normalize score in [0,1]
        score = min(max(total_reward / MAX_STEPS, 0.0), 1.0)
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] inference runtime error: {e}", flush=True)

    finally:
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception:
            pass

        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()
