"""
agent.py — D3QN Submission
CS780 OBELIX
"""

import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque

ACTIONS    = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS  = 5
OBS_DIM    = 18
FRAME_STACK= 4
STATE_DIM  = OBS_DIM * FRAME_STACK   # 72
HIDDEN1    = 256
HIDDEN2    = 128
VALUE_HIDDEN = 64
ADV_HIDDEN   = 64


class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN1), nn.ReLU(),
            nn.Linear(HIDDEN1,   HIDDEN2), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN2, VALUE_HIDDEN), nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(HIDDEN2, ADV_HIDDEN), nn.ReLU(),
            nn.Linear(ADV_HIDDEN, N_ACTIONS),
        )

    def forward(self, x):
        feat = self.backbone(x)
        V    = self.value_stream(feat)
        A    = self.adv_stream(feat)
        return V + (A - A.mean(dim=1, keepdim=True))


class FrameStacker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = deque(
            [np.zeros(OBS_DIM, dtype=np.float32)] * FRAME_STACK,
            maxlen=FRAME_STACK)

    def push(self, obs):
        self.frames.append(np.array(obs, dtype=np.float32))

    def get_state(self):
        return np.concatenate(list(self.frames))


_model   = None
_stacker = None


def _load():
    global _model, _stacker
    if _model is None:
        _model = DuelingDQN()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "d3qn_weights_backup.pth")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            _model.load_state_dict(ckpt["online"])
            print("D3QN weights loaded successfully.")
        _model.eval()
        _stacker = FrameStacker()


def policy(obs, rng=None) -> str:
    _load()
    _stacker.push(obs)
    state = torch.FloatTensor(
        _stacker.get_state()).unsqueeze(0)
    with torch.no_grad():
        idx = _model(state).argmax(dim=1).item()
    return ACTIONS[idx]


def reset_episode():
    _load()
    _stacker.reset()
