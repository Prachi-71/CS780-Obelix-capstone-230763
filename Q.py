"""
Q-Learning Training Script for OBELIX - ULTRA FAST OPTIMIZED
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
    print("ERROR: Run from the CS780-OBELIX repo root (where obelix.py lives).")
    sys.exit(1)

GAMMA         = 0.99
ALPHA_START   = 0.5
ALPHA_MIN     = 0.049
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
print(f"GAMMA: {GAMMA} | ALPHA: {ALPHA_START}->{ALPHA_MIN} | EPSILON: {EPSILON_START}->{EPSILON_MIN}")
NO_EPISODES   = 1000
max_steps = 500     

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
N_STATES  = 262144

Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)

POWERS_OF_TWO = 1 << np.arange(18)

def obs_to_state(obs):
    # this converts the obs from state from binary to integer
    return int(np.dot(obs, POWERS_OF_TWO))


def train():
    # diff 0 
    schedule = [
        (0, NO_EPISODES),        
    ]

    t_total = time.time()

    for difficulty, n_ep in schedule:
        label = {0: "Static Box", 2: "Blinking Box", 3: "Moving+Blinking"}[difficulty]
        print(f"\n{'='*55}")
        print(f"  {label}  (difficulty={difficulty}, {n_ep} eps)")
        print(f"{'='*55}")

        env = OBELIX(scaling_factor=1, difficulty=0, wall_obstacles=False, max_steps=500)
        rewards         = []
        successes       = 0
        t0              = time.time()
        t_total_level   = time.time()

        for ep in range(n_ep):
            alpha   = max(ALPHA_MIN,   ALPHA_START   * (ALPHA_MIN/ALPHA_START)   ** (ep / n_ep))
            epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_MIN/EPSILON_START) ** (ep / n_ep))

            obs  = env.reset()
            s    = obs_to_state(obs)
            done = False
            ep_r = 0.0
            step_count = 0

            while not done and step_count < 500:
                step_count += 1
                if random.random() < epsilon:
                    a = random.randint(0, 4)
                else:
                    a = int(Q[s].argmax())
                obs_next, r, done = env.step(ACTIONS[a], render=False)
                s_next = obs_to_state(obs_next)
                ep_r  += r
                td_target = r if done else r + GAMMA * Q[s_next].max()
                Q[s, a]  += alpha * (td_target - Q[s, a])
                s = s_next

            rewards.append(ep_r)
            if ep_r > 500:
                successes += 1

            
            if (ep + 1) % 50 == 0 or (ep + 1) == n_ep:
                elapsed_ep = time.time() - t0
                speed      = (ep + 1) / max(0.001, time.time() - t_total_level)
                remain     = (n_ep - ep - 1) / max(1.0, speed) / 60
                print(f"  ep {ep+1:4d}/{n_ep} | "
                      f"a={alpha:.3f} e={epsilon:.3f} | "
                      f"R={ep_r:8.1f} | "
                      f"AvgR={np.mean(rewards[-50:]):8.1f} | "
                      f"ok={successes:3d} | "
                      f"{speed:.0f} ep/s | "
                      f"~{remain:.1f} min left",
                      flush=True)

        print(f"  Level done. AvgR (last 2000): {np.mean(rewards[-2000:]):.1f}")
        

    # Save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(Q, f, protocol=4)

    print(f"\nTotal time : {(time.time()-t_total)/60:.1f} min")
    print(f"Q-table    : q_table.pkl  ({Q.nbytes//1024} KB on disk before compression)")
    return rewards


if __name__ == "__main__":
    rewards = train()
 
 
    w    = 50
    kern = np.ones(w) / w
    rewards = [float(r) for r in rewards if r is not None]
    avg  = np.convolve(rewards, kern, mode='valid')
    succ = np.convolve([1 if r > 500 else 0 for r in rewards], kern, mode='valid')
    eps  = np.arange(1, len(rewards) + 1)
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(eps, rewards, alpha=0.3, color='steelblue', label='Reward')
    ax1.plot(eps[w-1:], avg, color='navy', linewidth=2, label='Avg(50)')
    ax1.set(xlabel='Episode', ylabel='Reward', title='Q-Learning Reward vs Episodes')
    ax1.legend(); ax1.grid(True, alpha=0.3)
 
    ax2.plot(eps[w-1:], succ * 100, color='green', linewidth=2)
    ax2.set(xlabel='Episode', ylabel='Success %', title='Success Rate vs Episodes', ylim=(0,100))
    ax2.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig('training_plots.png', dpi=150)
    plt.show()
 
    for name, s in [("all-zero",0),("IR(16)",1<<16),("stuck(17)",1<<17),("fwd-near",1<<5),("left",1<<1),("right",1<<13)]:
        print(f"  {name:12s} -> {ACTIONS[int(np.argmax(Q[s]))]}")