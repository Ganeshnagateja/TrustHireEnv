import os
from typing import List, Optional, Tuple

from openai import OpenAI
from env.environment import TrustHireEnv

# ======================================================
# REQUIRED ENV VARIABLES (STRICT FOR SUBMISSION)
# ======================================================
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

BENCHMARK = "TrustHireEnv"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 5


# ======================================================
# REQUIRED LOGGING FORMAT
# ======================================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str]
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ======================================================
# LLM POLICY
# ======================================================
SYSTEM_PROMPT = """
You are controlling TrustHireEnv.

You must output EXACTLY one line in this format:
flag_level|next_step|rationale

Allowed flag_level values:
none
low
medium
high

Allowed next_step values:
continue
followup
warn
escalate

Choose the safest and most reward-improving action based on the observation and current step.
Do not output anything else.
""".strip()


def fallback_policy(task: str, step: int, observation: str) -> Tuple[str, str, str]:
    obs = str(observation).lower()

    critical_keywords = [
        "fraud", "fake", "forged", "tampered",
        "criminal", "blacklisted", "salary discrepancy",
        "identity mismatch"
    ]
    medium_keywords = [
        "missing", "gap", "unclear",
        "incomplete", "pending", "unverified"
    ]

    if any(k in obs for k in critical_keywords):
        return "high", "escalate", "High-risk verification failure detected"

    if any(k in obs for k in medium_keywords):
        if step <= 2:
            return "medium", "followup", "Additional verification required"
        return "high", "escalate", "Repeated unresolved issue"

    if task == "easy":
        if step == 1:
            return "low", "continue", "Initial profile consistency check"
        if step in [2, 3]:
            return "medium", "followup", "Running additional checks"
        if step == 4:
            return "high", "escalate", "Escalating after repeated checks"
        return "medium", "warn", "Final caution after escalation"

    if task == "medium":
        if step == 1:
            return "medium", "followup", "Medium-risk review started"
        if step == 2:
            return "medium", "followup", "Collecting further evidence"
        if step == 3:
            return "high", "warn", "Suspicion level increasing"
        if step == 4:
            return "high", "escalate", "Escalating unresolved anomalies"
        return "medium", "warn", "Final caution after escalation"

    # hard
    if step == 1:
        return "medium", "followup", "Hard-case verification initiated"
    if step == 2:
        return "high", "followup", "High-risk signals under review"
    if step == 3:
        return "high", "warn", "Escalation likely if inconsistencies remain"
    if step == 4:
        return "high", "escalate", "Escalating difficult unresolved case"
    return "medium", "warn", "Final caution after high-risk handling"


def parse_action(text: str) -> Optional[Tuple[str, str, str]]:
    text = (text or "").strip()
    parts = [p.strip() for p in text.split("|", 2)]
    if len(parts) != 3:
        return None

    flag_level, next_step, rationale = parts

    valid_flags = {"none", "low", "medium", "high"}
    valid_steps = {"continue", "followup", "warn", "escalate"}

    if flag_level not in valid_flags:
        return None
    if next_step not in valid_steps:
        return None
    if not rationale:
        rationale = "LLM-selected action"

    return flag_level, next_step, rationale


def choose_action_with_llm(
    client: OpenAI,
    task: str,
    step: int,
    observation
) -> Tuple[str, str, str]:
    obs_text = str(observation)

    user_prompt = f"""
Task: {task}
Step: {step}
Observation:
{obs_text}

Return exactly one line:
flag_level|next_step|rationale
""".strip()

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=40,
        )

        content = resp.choices[0].message.content or ""
        parsed = parse_action(content)
        if parsed is not None:
            return parsed

    except Exception:
        pass

    return fallback_policy(task, step, obs_text)


# ======================================================
# SCORING
# ======================================================
def score_from_rewards(rewards: List[float]) -> float:
    if not rewards:
        return 0.01

    best_reward = max(rewards)
    score = (best_reward + 1.0) / 2.0

    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    return score


# ======================================================
# RUN ONE TASK
# ======================================================
def run_task(client: OpenAI, task: str) -> None:
    env = TrustHireEnv(difficulty=task, seed=42)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset()

        for step in range(1, MAX_STEPS + 1):
            flag_level, next_step, rationale = choose_action_with_llm(
                client, task, step, observation
            )

            action_str = f"{flag_level}|{next_step}|{rationale}"

            observation, reward, done, info = env.step({
                "flag_level": flag_level,
                "next_step": next_step,
                "rationale": rationale,
            })

            reward = float(reward)
            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=bool(done),
                error=None,
            )

            if done:
                break

        score = score_from_rewards(rewards)
        success = score > 0.5

    except Exception as exc:
        print(f"[DEBUG] inference runtime error ({task}): {exc}", flush=True)
        score = 0.01
        success = False

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


# ======================================================
# MAIN
# ======================================================
def main():
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    for task in TASKS:
        run_task(client, task)


if __name__ == "__main__":
    main()
