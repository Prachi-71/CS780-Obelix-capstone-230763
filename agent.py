"""
agent.py — PPO Submission Wrapper for OBELIX (Codabench)
=========================================================
Compatible with evaluate.py calling: policy(obs, rng) -> action_str
                                      reset_episode()

Weight loading priority (first found wins):
  1. ppo_finetuned_2.pth
  2. ppo_tuned_final.pth
  3. ppo_finetuned.pth
  4. ppo_weights_bc.pth
  5. ppo_weights.pth
"""

import os
import random

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# ── Constants ────────────────────────────────────────────────────────────
ACTIONS       = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS     = 5
N_OBS         = 18
STACK_SIZE    = 4
N_OBS_STACKED = N_OBS * STACK_SIZE  # 72

WEIGHTS_PRIORITY = [
    "ppo_tuned_final.pth"
]

EVAL_EPSILON = 0.10   # Wanderer Fix: 10% random to escape Roomba Trap

# ── Optional heavy imports (failures won't block policy definition) ───────
try:
    import numpy as np
    import torch
    import torch.nn as nn
    from collections import deque
    _IMPORTS_OK = True
except Exception as _import_err:
    print(f"[agent.py] Import error: {_import_err}")
    _IMPORTS_OK = False


# ── Classes (only defined if imports OK) ─────────────────────────────────
if _IMPORTS_OK:

    class FrameStacker:
        def __init__(self):
            self.reset()

        def reset(self):
            self.frames = deque(
                [np.zeros(N_OBS, dtype=np.float32)] * STACK_SIZE,
                maxlen=STACK_SIZE,
            )

        def push(self, obs):
            self.frames.append(np.asarray(obs, dtype=np.float32))

        def get_state(self):
            return np.concatenate(list(self.frames))   # (72,)


    class ActorCritic(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128), nn.Tanh(),
                nn.Linear(128, 64),        nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1),
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 128), nn.Tanh(),
                nn.Linear(128, 64),        nn.Tanh(),
                nn.Linear(64, 1),
            )


    class PPOAgentWrapper:
        def __init__(self):
            self.device  = torch.device("cpu")
            self.policy  = ActorCritic(N_OBS_STACKED, N_ACTIONS).to(self.device)
            self.stacker = FrameStacker()
            self._loaded = False
            self._load()

        def _load(self):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            weights_path = None
            for fname in WEIGHTS_PRIORITY:
                candidate = os.path.join(current_dir, fname)
                if os.path.exists(candidate):
                    weights_path = candidate
                    break

            if weights_path is None:
                print(f"[agent.py] WARNING: No weights found — using random policy.")
                print(f"[agent.py] Searched: {WEIGHTS_PRIORITY}")
                return

            try:
                ckpt = torch.load(weights_path, map_location=self.device, weights_only=True)
                if "full" in ckpt:
                    self.policy.load_state_dict(ckpt["full"])
                elif "actor" in ckpt and "critic" in ckpt:
                    self.policy.actor.load_state_dict(ckpt["actor"])
                    self.policy.critic.load_state_dict(ckpt["critic"])
                else:
                    self.policy.load_state_dict(ckpt)
                self.policy.eval()
                self._loaded = True
                print(f"[agent.py] Loaded: {os.path.basename(weights_path)}")
            except Exception as e:
                print(f"[agent.py] ERROR loading weights: {e}")

        def act(self, obs):
            self.stacker.push(obs)
            state_t = torch.FloatTensor(
                self.stacker.get_state()).unsqueeze(0).to(self.device)

            with torch.no_grad():
                probs = self.policy.actor(state_t)
                if np.random.rand() < EVAL_EPSILON:
                    action_idx = np.random.randint(0, N_ACTIONS)
                else:
                    action_idx = torch.argmax(probs, dim=1).item()

            return ACTIONS[action_idx]

        def reset(self):
            self.stacker.reset()


# ── Singleton ─────────────────────────────────────────────────────────────
_agent = None

def _init_agent():
    """Lazy-initialize the agent singleton."""
    global _agent
    if _agent is not None:
        return
    if _IMPORTS_OK:
        try:
            _agent = PPOAgentWrapper()
        except Exception as e:
            print(f"[agent.py] Agent init error: {e}")
    else:
        print("[agent.py] Heavy imports unavailable — falling back to random policy.")


# ── PUBLIC API (always present at module level — evaluate.py checks these) ─

def policy(obs, rng=None):
    """Select an action given an observation. Required by evaluate.py."""
    _init_agent()
    if _agent is not None:
        return _agent.act(obs)
    # Emergency random fallback (no torch/numpy required)
    return random.choice(ACTIONS)


def reset_episode():
    """Reset per-episode state. Required by evaluate.py."""
    global _agent
    if _agent is not None:
        _agent.reset()