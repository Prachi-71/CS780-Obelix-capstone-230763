import os
import random
from collections import deque

# Suppress warnings and GUI popups
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

# --- Configuration ---
WALLS_ENABLED = True 

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = len(ACTIONS)

# Sensor Layout (18 total inputs)
N_OBS = 18
STACK_SIZE = 4
N_OBS_STACKED = N_OBS * STACK_SIZE 

# Sensor index groups (mapping binary bits to logical positions)
S_FAR = [0, 2, 4, 6, 8, 10, 12, 14]
S_NEAR = [1, 3, 5, 7, 9, 11, 13, 15]
IR_IDX = 16

# Logical sensor groups
LEFT_BACK_FAR = [0, 2]
ALL_LEFT_FAR = [0, 2, 4, 6]
ALL_RIGHT_FAR = [8, 10, 12, 14]
RIGHT_BACK_FAR = [12, 14]
FORWARD_FAR = [6, 8]
FORWARD_NEAR = [7, 9]

# Thresholds for classification
BOX_MAX_SENSORS = 3
WALL_MIN_SENSORS = 4
GAP_SILENT_THRESH = 0
PARALLEL_MAX_STEPS = 25

# Oscillation detection (Roomba trap)
ROOMBA_HISTORY = 12
ROOMBA_PATTERNS = [
    (["L45", "R45"] * 6)[-ROOMBA_HISTORY:],
    (["R45", "L45"] * 6)[-ROOMBA_HISTORY:],
    (["L22", "R22"] * 6)[-ROOMBA_HISTORY:],
    (["R22", "L22"] * 6)[-ROOMBA_HISTORY:],
    (["FW", "L45"] * 6)[-ROOMBA_HISTORY:],
    (["FW", "R45"] * 6)[-ROOMBA_HISTORY:],
]

# PPO weights priority
PPO_WEIGHTS = [
    "ppo_finetuned_2.pth",
    "ppo_tuned_final.pth",
    "ppo_finetuned.pth",
    "ppo_weights_bc.pth",
    "ppo_weights.pth",
]

try:
    import numpy as np
    import torch
    import torch.nn as nn
    _IMPORTS_OK = True
except Exception as e:
    print(f"Loading error: {e}")
    _IMPORTS_OK = False

if _IMPORTS_OK:
    class FrameStacker:
        """Maintains the 4-frame temporal context for the neural network"""
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
            return np.concatenate(list(self.frames))

    class ActorCritic(nn.Module):
        def __init__(self, state_dim=N_OBS_STACKED, action_dim=N_ACTIONS):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128), nn.Tanh(),
                nn.Linear(128, 64), nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1),
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 128), nn.Tanh(),
                nn.Linear(128, 64), nn.Tanh(),
                nn.Linear(64, 1),
            )

    class PPOCore:
        """Handles PPO model loading and inference"""
        def __init__(self):
            self.device = torch.device("cpu")
            self.net = ActorCritic().to(self.device)
            self.stacker = FrameStacker()
            self.loaded = False
            self._load_weights()

        def _load_weights(self):
            base_path = os.path.dirname(os.path.abspath(__file__))
            for f_name in PPO_WEIGHTS:
                path = os.path.join(base_path, f_name)
                if os.path.exists(path):
                    try:
                        ckpt = torch.load(path, map_location=self.device, weights_only=True)
                        if isinstance(ckpt, dict):
                            if "full" in ckpt: self.net.load_state_dict(ckpt["full"])
                            elif "actor" in ckpt: self.net.actor.load_state_dict(ckpt["actor"])
                            else: self.net.load_state_dict(ckpt)
                        else:
                            self.net.load_state_dict(ckpt)
                        self.net.eval()
                        self.loaded = True
                        print(f"PPO Weights Loaded: {f_name}")
                        return
                    except Exception as e:
                        print(f"Error loading {f_name}: {e}")
            print("PPO: No weights found. Running with random initialization.")

        def act(self, obs):
            self.stacker.push(obs)
            state = torch.FloatTensor(self.stacker.get_state()).unsqueeze(0)
            with torch.no_grad():
                probs = self.net.actor(state)
                choice = torch.argmax(probs, dim=1).item()
            return ACTIONS[choice]

        def reset(self):
            self.stacker.reset()

class SARLayer:
    """Heuristic override for wall navigation and obstacle classification"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._history = deque(maxlen=ROOMBA_HISTORY)
        self._parallel_side = None
        self._parallel_steps = 0
        self._prev_obs = None

    def decide(self, obs, ppo_action):
        obs_list = list(obs)

        # Break out of oscillation loops
        escape_move = self._check_roomba()
        if escape_move:
            self._log(escape_move, obs_list)
            return escape_move

        # IR contact sensor check
        if int(obs_list[IR_IDX]):
            self._log(ppo_action, obs_list)
            return ppo_action

        # Check if forward path is blocked
        fwd_blocked = sum(1 for i in FORWARD_FAR + FORWARD_NEAR if obs_list[i] == 1) > 0

        if fwd_blocked:
            obj_type = self._classify(obs_list)
            
            if obj_type == "BOX":
                final_action = ppo_action # Trust PPO to approach the box
            else:
                final_action = self._parallel_travel(obs_list)
        elif self._parallel_side is not None:
            final_action = self._find_gap(obs_list)
        else:
            final_action = ppo_action

        self._log(final_action, obs_list)
        return final_action

    def _classify(self, obs):
        if not WALLS_ENABLED: return "BOX"
        
        total_far = sum(1 for i in S_FAR if obs[i] == 1)

        # Basic size classification
        if total_far <= BOX_MAX_SENSORS:
            # Check for flanking wall
            if sum(1 for i in LEFT_BACK_FAR + RIGHT_BACK_FAR if obs[i] == 1) > 0:
                return "WALL"
            return "BOX"

        if total_far >= WALL_MIN_SENSORS:
            return "WALL"

        # Check for sudden appearance (Box-like behavior)
        if self._prev_obs:
            prev_fwd = sum(1 for i in FORWARD_FAR if self._prev_obs[i] == 1)
            curr_near = sum(1 for i in FORWARD_NEAR if obs[i] == 1)
            if prev_fwd == 0 and curr_near > 0:
                return "BOX"

        return "WALL"

    def _parallel_travel(self, obs):
        if self._parallel_side is None:
            l_hit = sum(1 for i in ALL_LEFT_FAR if obs[i] == 1)
            r_hit = sum(1 for i in ALL_RIGHT_FAR if obs[i] == 1)
            self._parallel_side = "LEFT" if l_hit <= r_hit else "RIGHT"
            self._parallel_steps = 0
        
        self._parallel_steps += 1
        if self._parallel_steps > PARALLEL_MAX_STEPS:
            self._parallel_side = None
            return random.choice(["L45", "R45"])

        return "L22" if self._parallel_side == "LEFT" else "R22"

    def _find_gap(self, obs):
        side_idx = ALL_LEFT_FAR if self._parallel_side == "LEFT" else ALL_RIGHT_FAR
        turn_in = "L45" if self._parallel_side == "LEFT" else "R45"
        keep_going = "L22" if self._parallel_side == "LEFT" else "R22"

        if sum(1 for i in side_idx if obs[i] == 1) <= GAP_SILENT_THRESH:
            # confirm path exists or enough steps taken
            if sum(1 for i in FORWARD_FAR if obs[i] == 1) > 0 or self._parallel_steps >= 5:
                self._parallel_side = None
                return turn_in
            
        self._parallel_steps += 1
        return keep_going if self._parallel_steps <= PARALLEL_MAX_STEPS else "FW"

    def _check_roomba(self):
        h = list(self._history)
        if len(h) < ROOMBA_HISTORY: return None
        for pattern in ROOMBA_PATTERNS:
            if h == pattern:
                return random.choice([a for a in ACTIONS if a != h[-1]])
        return None

    def _log(self, action, obs):
        self._history.append(action)
        self._prev_obs = obs

class HybridPPOAgent:
    def __init__(self):
        self.sar = SARLayer()
        self.ppo = PPOCore() if _IMPORTS_OK else None
        print(f"Hybrid Agent Initialized. Walls: {WALLS_ENABLED}")

    def act(self, obs):
        ppo_action = self.ppo.act(obs) if self.ppo else random.choice(ACTIONS)
        return self.sar.decide(obs, ppo_action)

    def reset(self):
        self.sar.reset()
        if self.ppo: self.ppo.reset()

# --- Public Interface for evaluate.py ---
_agent = None

def policy(obs, rng=None):
    global _agent
    if _agent is None:
        _agent = HybridPPOAgent()
    return _agent.act(obs)

def reset_episode():
    global _agent
    if _agent is not None:
        _agent.reset()