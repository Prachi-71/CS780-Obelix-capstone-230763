"""
DYNA-Q Training Script for OBELIX Warehouse Robot
CS780 Capstone Project - Level 1 (Static Box)

Implements DYNA-Q exactly as per CS780 lecture pseudocode
(Prof. Ashutosh Modi, Slide 56-60)

Algorithm:
  Initialize Q[s,a]=0, T[s,a,s']=0, R[s,a,s']=0
  for e in range(noEpisodes):
      alpha = decayLearningRate(e)
      epsilon = decayEpsilon(e)
      s, done = env.reset()
      while not done:
          a = actionSelect(s, Q, epsilon)
          s', r, done = env.step(a)
          T[s,a,s'] += 1
          R[s,a,s'] += r
          td_target = r [+ gamma * Q[s'].max() if not done]
          td_error = td_target - Q[s,a]
          Q[s,a] = Q[s,a] + alpha * td_error
          s_backUp = s
          for _ in range(noPlanning):
              if sum(Q) == 0: break
              s = random.choice(s_visited)
              a = random.choice(a_taken[s])
              probs_a = T[s,a] / sum(T[s,a,:])
              s' = random.choice(S, 1, probs_a)
              r = R[s,a,s'] / T[s,a,s']
              td_target = r + gamma * Q[s'].max()
              td_error = td_target - Q[s,a]
              Q[s,a] = Q[s,a] + alpha * td_error
          s = s_backUp
"""

import numpy as np
import pickle
import sys
import time
import random
import matplotlib.pyplot as plt

try:
    from obelix import OBELIX
except ImportError:
    print("ERROR: Run from CS780-OBELIX repo root.")
    sys.exit(1)

# -----------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------
GAMMA         = 0.99
ALPHA_START   = 0.5
ALPHA_MIN     = 0.05
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
NO_EPISODES   = 1000
MAX_STEPS     = 1000
NO_PLANNING   = 10      # K planning steps per real step

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
N_STATES  = 1 << 18

print(f"GAMMA={GAMMA} | ALPHA={ALPHA_START}->{ALPHA_MIN} | "
      f"EPSILON={EPSILON_START}->{EPSILON_MIN} | K={NO_PLANNING}")

# -----------------------------------------------------------------------
# Q-table
# -----------------------------------------------------------------------
Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)
T = {}   # transition counts
R = {}   # reward sums

# Track visited states and actions taken at each state
s_visited = []
a_taken   = {}

POWERS = (1 << np.arange(18, dtype=np.int64))

def obs_to_state(obs):
    return int(np.dot(obs.astype(np.int64), POWERS))

def decay_learning_rate(episode):
    frac = episode / max(1, NO_EPISODES)
    return max(ALPHA_MIN, ALPHA_START * (1.0 - frac))

def decay_epsilon(episode):
    frac = episode / max(1, NO_EPISODES)
    return max(EPSILON_MIN, EPSILON_START * (1.0 - 0.9 * frac))

def action_select(s, Q, epsilon):
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    return int(np.argmax(Q[s]))

# -----------------------------------------------------------------------
# Training loop — matches lecture pseudocode exactly
# -----------------------------------------------------------------------
def train():
    rewards   = []
    successes = 0
    t_total   = time.time()

    env = OBELIX(scaling_factor=1, difficulty=0,
                 wall_obstacles=False, max_steps=MAX_STEPS)

    print(f"\n{'='*60}")
    print(f"  DYNA-Q | Static Box | {NO_EPISODES} eps | K={NO_PLANNING}")
    print(f"{'='*60}")

    for e in range(NO_EPISODES):
        alpha   = decay_learning_rate(e)
        epsilon = decay_epsilon(e)

        obs  = np.asarray(env.reset(), dtype=np.int64)
        s    = obs_to_state(obs)
        done = False
        ep_r = 0.0
        step = 0

        while not done and step < MAX_STEPS:
            step += 1

            # --- Action selection (epsilon-greedy) ---
            a = action_select(s, Q, epsilon)

            # --- Real environment step ---
            obs_next, r, done = env.step(ACTIONS[a], render=False)
            obs_next = np.asarray(obs_next, dtype=np.int64)
            s_next   = obs_to_state(obs_next)
            ep_r    += r

            # --- Model update: T[s,a,s'] and R[s,a,s'] ---
            if s not in T:
                T[s] = {}
                R[s] = {}
            if a not in T[s]:
                T[s][a] = {}
                R[s][a] = {}
            T[s][a][s_next]  = T[s][a].get(s_next, 0) + 1
            R[s][a][s_next]  = R[s][a].get(s_next, 0.0) + r

            # Track visited states and actions
            if s not in a_taken:
                s_visited.append(s)
                a_taken[s] = []
            if a not in a_taken[s]:
                a_taken[s].append(a)

            # --- Direct Q-update from real experience ---
            td_target = r if done else r + GAMMA * float(Q[s_next].max())
            td_error  = td_target - Q[s, a]
            Q[s, a]  += alpha * td_error

            # --- Planning: K simulated updates ---
            s_backUp = s

            for _ in range(NO_PLANNING):
                if not s_visited:
                    break

                # Sample random visited state and action
                ps = random.choice(s_visited)
                pa = random.choice(a_taken[ps])

                # Get transition probabilities from model
                t_counts  = T[ps][pa]
                total     = sum(t_counts.values())
                states_p  = list(t_counts.keys())
                probs     = np.array([t_counts[sp] / total for sp in states_p])

                # Sample next state from model
                ps_next = states_p[np.random.choice(len(states_p), p=probs)]

                # Get expected reward from model
                pr = R[ps][pa][ps_next] / T[ps][pa][ps_next]

                # Planning Q-update
                td_plan  = pr + GAMMA * float(Q[ps_next].max())
                td_error = td_plan - Q[ps, pa]
                Q[ps, pa] += alpha * td_error

            s = s_backUp  # restore state after planning
            s = s_next    # advance to real next state

        rewards.append(ep_r)
        if ep_r > 500:
            successes += 1

        if (e + 1) % 50 == 0 or (e + 1) == NO_EPISODES:
            elapsed = time.time() - t_total
            speed   = (e + 1) / max(0.001, elapsed)
            remain  = (NO_EPISODES - e - 1) / max(0.001, speed) / 60
            print(f"  ep {e+1:4d}/{NO_EPISODES} | "
                  f"a={alpha:.3f} e={epsilon:.3f} | "
                  f"R={ep_r:8.1f} | "
                  f"AvgR={np.mean(rewards[-50:]):8.1f} | "
                  f"ok={successes:3d} | "
                  f"model={len(s_visited):4d} | "
                  f"{speed:.2f} ep/s | "
                  f"~{remain:.1f} min left",
                  flush=True)

    # Save Q-table (same format as Q-learning, same agent.py works)
    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q, f, protocol=4)

    print(f"\nTotal time : {(time.time()-t_total)/60:.1f} min")
    print(f"Model size : {len(s_visited)} unique states learned")
    print(f"Q-table    : q_table.pkl ({Q.nbytes//1024} KB)")
    return rewards


if __name__ == "__main__":
    rewards = train()

    print("\n--- Greedy policy check ---")
    checks = [
        ("all-zero",  0),
        ("IR(16)",    1 << 16),
        ("stuck(17)", 1 << 17),
        ("fwd-near",  1 << 5),
        ("left(1)",   1 << 1),
        ("right(13)", 1 << 13),
    ]
    for name, s in checks:
        print(f"  {name:12s} -> {ACTIONS[int(np.argmax(Q[s]))]}")

    w    = 50
    kern = np.ones(w) / w
    rewards = [float(r) for r in rewards if r is not None]
    avg  = np.convolve(rewards, kern, mode='valid')
    succ = np.convolve([1 if r > 500 else 0 for r in rewards], kern, mode='valid')
    eps  = np.arange(1, len(rewards) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(eps, rewards, alpha=0.3, color='steelblue', label='Reward')
    ax1.plot(eps[w-1:], avg, color='navy', linewidth=2, label='Avg(50)')
    ax1.set(xlabel='Episode', ylabel='Reward', title='DYNA-Q Reward vs Episodes')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(eps[w-1:], succ * 100, color='green', linewidth=2)
    ax2.set(xlabel='Episode', ylabel='Success %',
            title='Success Rate vs Episodes', ylim=(0, 100))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dynaq_plots.png', dpi=150)
    plt.show()
