"""
agent.py — PPO Submission Interface
=====================================
Drop this file + ppo_finetuned.pth in your submission folder.
Rename ppo_finetuned.pth to ppo_weights.pth if your evaluator expects that name.

Compatible with the ActorCritic architecture from ppo.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.distributions import Categorical

ACTIONS       = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS     = 5
N_OBS         = 18
STACK_SIZE    = 4
N_OBS_STACKED = N_OBS * STACK_SIZE  # 72


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


_model   = None
_obs_queue = None


def _load():
    global _model, _obs_queue
    if _model is None:
        _model = ActorCritic(N_OBS_STACKED, N_ACTIONS)

        # Try finetuned first, fall back to original
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for fname in ["ppo_finetuned.pth", "ppo_weights.pth"]:
            path = os.path.join(current_dir, fname)
            if os.path.exists(path):
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
                if "full" in ckpt:
                    _model.load_state_dict(ckpt["full"])
                elif "actor" in ckpt and "critic" in ckpt:
                    _model.actor.load_state_dict(ckpt["actor"])
                    _model.critic.load_state_dict(ckpt["critic"])
                else:
                    _model.load_state_dict(ckpt)
                print(f"PPO agent loaded from: {fname}")
                break

        _model.eval()
        _obs_queue = deque(
            [np.zeros(N_OBS, dtype=np.float32)] * STACK_SIZE,
            maxlen=STACK_SIZE
        )


def policy(obs, rng=None) -> str:
    _load()
    _obs_queue.append(np.asarray(obs, dtype=np.float32))
    state = np.concatenate(list(_obs_queue))

    with torch.no_grad():
        probs = _model.actor(torch.FloatTensor(state).unsqueeze(0))
        idx   = probs.argmax(dim=1).item()   # greedy at eval time

    return ACTIONS[idx]


def reset_episode():
    """Called by evaluator at the start of each episode."""
    _load()
    for i in range(STACK_SIZE):
        _obs_queue[i] = np.zeros(N_OBS, dtype=np.float32)
