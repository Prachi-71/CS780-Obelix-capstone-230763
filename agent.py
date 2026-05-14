import os
import random
from collections import deque

# Headless environment setup
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# --- Constants ---
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
S_FAR = [0, 2, 4, 6, 8, 10, 12, 14]
FORWARD_SENSORS = [6, 7, 8, 9] # S3/S4 Far and Near bits

try:
    import numpy as np
    import torch
    import torch.nn as nn
    _IMPORTS_OK = True
except Exception:
    _IMPORTS_OK = False

# --- PPO Model Architecture ---
if _IMPORTS_OK:
    class ActorCritic(nn.Module):
        def __init__(self, state_dim=72, action_dim=5):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128), nn.Tanh(),
                nn.Linear(128, 64), nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

    class PPOInference:
        def __init__(self):
            self.net = ActorCritic()
            self.stacker = deque(maxlen=4)
            self._load_weights()

        def _load_weights(self):
            """Robust loader to handle the state_dict key mismatch"""
            for f in ["ppo_tuned_final.pth"]:
                if not os.path.exists(f):
                    continue
                try:
                    ckpt = torch.load(f, map_location="cpu", weights_only=True)
                    
                    # Extract the dictionary if it's a full checkpoint
                    state_dict = ckpt["actor"] if (isinstance(ckpt, dict) and "actor" in ckpt) else ckpt
                    
                    # Fix "0.weight" vs "actor.0.weight" mismatch
                    first_key = list(state_dict.keys())[0]
                    if first_key.startswith("0.") or first_key.startswith("2."):
                        state_dict = {f"actor.{k}": v for k, v in state_dict.items()}

                    self.net.load_state_dict(state_dict, strict=False)
                    self.net.eval()
                    print(f"Agent weights loaded successfully: {f}")
                    return
                except Exception as e:
                    print(f"Could not load {f}: {e}")

        def get_action(self, obs):
            if not _IMPORTS_OK: return 2 # Default to Forward
            # Initialize stacker if empty
            if len(self.stacker) == 0:
                for _ in range(4): self.stacker.append(obs)
            self.stacker.append(obs)
            
            # Prepare tensor
            state = torch.FloatTensor(np.concatenate(list(self.stacker))).unsqueeze(0)
            with torch.no_grad():
                probs = self.net.actor(state)
                return torch.argmax(probs, dim=1).item()

class OBELIXAgent:
    def __init__(self):
        self.ppo = PPOInference() if _IMPORTS_OK else None
        self.parallel_side = None
        self.is_avoiding_wall = False

    def _classify_object(self, obs):
        """
        Differentiates objects based on sensor firing density.
        RL agents often struggle with aliasing (box vs wall looking same).
        """
        active_far = sum(obs[i] for i in S_FAR)
        # 1-3 sensors firing = discrete object (Box)
        # 4+ sensors firing = continuous surface (Wall)
        return "WALL" if active_far >= 4 else "BOX"

    def act(self, obs):
        # 1. Get the primary intent from the PPO network
        ppo_idx = self.ppo.get_action(obs) if self.ppo else 2
        ppo_action = ACTIONS[ppo_idx]
        
        # 2. Analyze environment state
        obj_type = self._classify_object(obs)
        is_blocked = any(obs[i] == 1 for i in FORWARD_SENSORS)

        # 3. Decision Logic:
        # If the PPO sees a BOX, we let the RL stay in 100% control.
        if obj_type == "BOX":
            self.is_avoiding_wall = False
            self.parallel_side = None
            return ppo_action

        # If it's a WALL and we're blocked, we trigger a safety parallel nudge.
        if obj_type == "WALL" and is_blocked:
            self.is_avoiding_wall = True
            if self.parallel_side is None:
                # Move toward the side with fewer active sensors (cleaner path)
                self.parallel_side = "LEFT" if sum(obs[0:6]) < sum(obs[10:16]) else "RIGHT"
            return "L22" if self.parallel_side == "LEFT" else "R22"

        # If the wall is no longer blocking our front, turn back in to find the gap.
        if self.is_avoiding_wall and not is_blocked:
            self.is_avoiding_wall = False
            prev_side = self.parallel_side
            self.parallel_side = None
            return "L45" if prev_side == "LEFT" else "R45"

        # Otherwise, trust the trained PPO behavior
        return ppo_action

    def reset(self):
        if self.ppo:
            self.ppo.stacker.clear()
        self.parallel_side = None
        self.is_avoiding_wall = False

# --- Codabench/Evaluation Hooks ---
_agent_instance = OBELIXAgent()

def policy(obs, rng=None):
    return _agent_instance.act(obs)

def reset_episode():
    _agent_instance.reset()