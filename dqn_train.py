import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import datetime
from collections import deque
import hex_engine as engine
import os
import submission.dqn_config as config
from submission.dqn_model import DuelingDQN as DQN #import DQN
from submission.dqn_utils import board_to_tensor, copy_game
from evaluation_positions import get_test_positions
from submission.base_agents import (
    random_agent,
    greedy_center_agent,
    edge_agent,
    avoid_center_agent,
    center_loving_agent,
    blocking_agent,
    greedy_connection_agent,
    first_available_agent,
    diagonal_agent,
    mirror_agent
)
import math

# Define the base agents list
base_agents_list = [
    random_agent,
    greedy_center_agent,
    edge_agent,
    avoid_center_agent,
    center_loving_agent,
    blocking_agent,
    greedy_connection_agent,
    first_available_agent,
    diagonal_agent,
    mirror_agent
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 100000
TARGET_UPDATE = 10
MEMORY_SIZE = 50000
STEP_OPTIMIZER_COUNT = 0     # global step counter for hard updates

# for running reward normalization
class RunningNorm:
    def __init__(self, eps=1e-2):
        self.mean = 0.0
        self.sq_mean = 0.0
        self.count = 0
        self.eps = eps
    def update(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.sq_mean += delta * (x - self.mean)
    def std(self):
        return math.sqrt(self.sq_mean / self.count + self.eps)
    def normalize(self, x):
        return (x - self.mean) / self.std()

# for n-step returns
class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = deque()
    def push(self, transition):
        """
        transition: (state, prev_player, action, reward, next_state, done)
        Returns a 5-tuple (state, action, R, next_state, done) once we have n steps; otherwise returns None.
        """
        self.buffer.append(transition)
        if len(self.buffer) < self.n:
            return None

        # compute raw n-step return from the un-normalized immediate rewards
        R_raw = 0.0
        for i, (_, _, _, r_i, _, _) in enumerate(self.buffer):
            R_raw += (self.gamma**i) * r_i
        # clip then normalize
        R_clipped = max(-1.0, min(1.0, R_raw))

        # unpack first and last
        state, prev_player, action, _, _, _ = self.buffer[0]
        _, _, _, _, next_state, done = self.buffer[-1]

        # pop the oldest and return
        self.buffer.popleft()
        return (state, prev_player, action, R_clipped, next_state, done)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

def select_action(game_state, policy_net, action_set, steps_done, eps_threshold):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            state_tensor = board_to_tensor(game_state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor).squeeze(0)
            
            # Create mask for valid actions
            mask = torch.zeros_like(q_values, dtype=torch.bool)
            for action in action_set:
                scalar = game_state.coordinate_to_scalar(action)
                mask[scalar] = True
            
            # Apply mask - invalid actions get min value
            masked_q = q_values.clone()
            masked_q[~mask] = -float('inf')
            
            # Select action with highest Q-value
            scalar_action = torch.argmax(masked_q).item()
            return game_state.scalar_to_coordinates(scalar_action), scalar_action
    else:
        action = random.choice(action_set)
        scalar = game_state.coordinate_to_scalar(action)
        return action, scalar

def optimize_model(memory, policy_net, target_net, optimizer):
    global STEP_OPTIMIZER_COUNT
    if len(memory) < BATCH_SIZE:
        return 0.0
    
    STEP_OPTIMIZER_COUNT += 1
    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))
    
    # Filter out None states
    non_final_mask = torch.tensor([s is not None for s in batch[3]], dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch[3] if s is not None]).to(device)
    
    state_batch = torch.stack(batch[0]).to(device)
    action_batch = torch.tensor(batch[1], dtype=torch.long, device=device)
    reward_batch = torch.tensor(batch[2], dtype=torch.float, device=device)
    done_batch = torch.tensor(batch[4], dtype=torch.float, device=device)
    
    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    
    # Compute V(s_{t+1}) for all next states
    # next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # with torch.no_grad():
    #     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Double-DQN: get argmax next from policy_net, but evaluate with target_net
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # 1) actions with highest Q under policy_net
        next_policy_q = policy_net(non_final_next_states)
        next_actions = next_policy_q.argmax(dim=1, keepdim=True)
        # 2) evaluate those actions in target_net
        next_target_q = target_net(non_final_next_states)
        chosen_q = next_target_q.gather(1, next_actions).squeeze(1)
        next_state_values[non_final_mask] = chosen_q

    # Compute the expected Q values
    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
    
    # Compute loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping - critical for stability
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), config.GRAD_CLIP)
    
    optimizer.step()

    # # hard‐update of target_net every TARGET_HARD_UPDATE_INTERVAL steps
    # if STEP_OPTIMIZER_COUNT % config.TARGET_HARD_UPDATE_INTERVAL == 0:
    #     target_net.load_state_dict(policy_net.state_dict())

    return loss.item()

def evaluate_against_base_agents(net, base_agents, num_games=20):  # Increased from 5 to 20
    results = {}
    net.eval()
    
    for agent_fn in base_agents:
        wins = 0
        for i in range(num_games):
            game = engine.hexPosition(size=config.BOARD_SIZE)
            starts_first = (i % 2 == 0)
            
            while game.winner == 0:
                if (game.player == 1 and starts_first) or (game.player == -1 and not starts_first):
                    # Use DQN
                    state_tensor = board_to_tensor(game).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = net(state_tensor).squeeze(0)
                    
                    # Mask invalid actions
                    mask = torch.zeros_like(q_values, dtype=torch.bool)
                    action_set = game.get_action_space()
                    for action in action_set:
                        scalar = game.coordinate_to_scalar(action)
                        mask[scalar] = True
                    
                    masked_q = q_values.clone()
                    masked_q[~mask] = -float('inf')
                    scalar_action = torch.argmax(masked_q).item()
                    action = game.scalar_to_coordinates(scalar_action)
                else:
                    # Use base agent
                    action = agent_fn(game.board, game.get_action_space())
                
                game.move(action)
            
            # Determine winner from DQN's perspective
            if (starts_first and game.winner == 1) or (not starts_first and game.winner == -1):
                wins += 1
        
        results[agent_fn.__name__] = wins / num_games
    
    net.train()
    return results

def soft_update(target_net, policy_net, tau):
    """Soft update target network parameters"""
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

def main():
    nstep_buffer = NStepBuffer(n=3, gamma=GAMMA)    # 3-step returns
    board_size = config.BOARD_SIZE
    action_size = board_size * board_size
    policy_net = DQN(board_size, action_size).to(device)
    target_net = DQN(board_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # optimizer = optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
    optimizer = optim.AdamW(
        policy_net.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY          # small decay on all weights
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPISODES)
    memory = ReplayMemory(MEMORY_SIZE)

    # Logging metrics
    episode_rewards = []
    win_rates = deque(maxlen=100)
    losses = deque(maxlen=100)
    start_time = time.time()
    
    steps_done = 0
    ret_norm = RunningNorm()
    for episode in range(config.NUM_EPISODES):
        # Determine self-play ratio
        selfplay_ratio = 0.15 + 0.35 * min(1.0, episode / (config.NUM_EPISODES * 0.7))
        use_selfplay = random.random() < selfplay_ratio
        opponent = None if use_selfplay else random.choice(base_agents_list)
        nstep_buffer.buffer.clear()
        
        # Randomly assign DQN to play as white (1) or black (-1) against base agents
        dqn_player = random.choice([1, -1]) if not use_selfplay else None
        
        game = engine.hexPosition(size=board_size)
        total_reward = 0
        done = False
        
        # Negative reward for longer games
        move_count = 0
        max_moves = board_size * board_size
        base_step_penalty = -0.05
        
        while not done:
            move_count += 1
            action_set = game.get_action_space()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            
            # DQN's turn (in self-play or assigned player role)
            if use_selfplay or game.player == dqn_player:
                state_tensor = board_to_tensor(game).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = policy_net(state_tensor).squeeze(0)
                
                # Mask invalid actions
                mask = torch.zeros_like(q_values, dtype=torch.bool)
                for action in action_set:
                    scalar = game.coordinate_to_scalar(action)
                    mask[scalar] = True
                
                masked_q = q_values.clone()
                masked_q[~mask] = -float('inf')
                scalar_action = torch.argmax(masked_q).item()
                action = game.scalar_to_coordinates(scalar_action)
            else:
                # Base agent's turn
                action = opponent(game.board, action_set)
                scalar_action = game.coordinate_to_scalar(action)
            
            # Store before state
            prev_state = board_to_tensor(game)
            prev_player = game.player
            game.move(action)
            
            # Calculate reward
            # reward = step_penalty  # Penalty for each move made
            length_ratio = move_count / max_moves
            # reward = base_step_penalty * (1.0 + length_ratio)  # e.g. up to −0.06 near end
            done = False
            reward = config.MOVE_PENALTY # Apply move penalty for every non-terminal move
            
            if game.winner != 0:
                # Game ended - determine final reward from DQN's perspective
                if use_selfplay:
                    # we need to know which “side” made this move:
                    # assume prev_player was stored alongside prev_state
                    if game.winner == prev_player:
                        reward = 1
                    else:
                        reward = -1
                else:
                    # Against base agent: reward based on DQN's role
                    if (dqn_player == 1 and game.winner == 1) or (dqn_player == -1 and game.winner == -1):
                        reward += 1  # Win reward
                    else:
                        reward -= 1  # Loss penalty
                if reward > 0:  # If DQN won
                    # Bonus for winning quickly: higher for shorter games
                    win_efficiency_bonus = (1.0 - (move_count / max_moves)) * config.WIN_EFFICIENCY_BONUS_SCALE
                    reward += win_efficiency_bonus
                # No explicit penalty for losing slowly, as -1 plus move_penalty handles it.
                
                # # Additional reward for winning quickly
                # if reward > 0:  # Only for wins
                #     win_bonus = (1 - move_count / max_moves) * 0.5  # Up to 0.5 extra reward
                #     reward += win_bonus
                
                done = True
                next_state = None
            else:
                next_state = board_to_tensor(game)

            # # ——— encourage straighter progress ——
            # # In Hex, White (player=1) aims top←→bottom, Black (player=-1) left←→right.
            # # Reward moves that advance along your axis.
            # if game.player == 1:
            #     # white just moved to next_state, r,c index of last move:
            #     r,c = action
            #     # closer to bottom edge is better (bigger r)
            #     reward += (r / (board_size-1)) * 0.02
            # elif game.player == -1:
            #     # black: closer to right edge is better (bigger c)
            #     r,c = action
            #     reward += (c / (board_size-1)) * 0.02

            # # ——— New: extra bonus for winning faster ——
            # # scale by fraction of empty board remaining
            # remaining = max_moves - move_count
            # reward += (remaining / max_moves) * 0.15
            
            # —— reward clipping & normalization —— 
            # clip to [-1, +1]
            #reward = max(-1.0, min(1.0, reward))
            # update running stats on n-step returns before push
            # ret_norm.update(reward)
            # reward = ret_norm.normalize(reward)

            total_reward += reward
            
            # Only store transitions for DQN's moves
            is_dqn_move = use_selfplay or (game.player != prev_player and not use_selfplay)
            if is_dqn_move:
                # memory.push(
                #     prev_state,
                #     scalar_action,
                #     reward,
                #     next_state,
                #     done
                # )
                # push to n-step buffer first
                tri = (prev_state, prev_player, scalar_action, reward, next_state, done)
                ns_ret = nstep_buffer.push(tri)
                if ns_ret is not None:
                    # memory.push(*ns_ret)

                    # ns_ret = (state, prev_player, action, R_clipped, next_state, done)
                    # normalize the raw R before storing
                    _, _, a, R_raw, s1, d = ns_ret
                    # Use the already clipped reward R_clipped from NStepBuffer
                    ret_norm.update(R_raw) # Note: R_raw here is R_clipped from NStepBuffer
                    R_norm = ret_norm.normalize(R_raw) # Normalize the clipped value
                    memory.push(ns_ret[0], ns_ret[2], R_norm, ns_ret[4], ns_ret[5])
            
            # Optimize model and track loss
            if len(memory) > BATCH_SIZE:
                loss_value = optimize_model(memory, policy_net, target_net, optimizer)
                scheduler.step()
                losses.append(loss_value)
                
                # Soft update of the target network - CRITICAL FIX
                soft_update(target_net, policy_net, config.TAU)
            
            steps_done += 1

        # flush any remaining partial returns (length < n)
        # for k = 1 .. len(buffer)-1, emit k-step returns
        leftover = list(nstep_buffer.buffer)
        for k in range(1, len(leftover)+1):
            seq = leftover[:k]
            # compute raw k-step return
            R_raw = sum((GAMMA**i)*seq[i][3] for i in range(len(seq)))
            R_clipped = R_raw# max(-1.0, min(1.0, R_raw))
            # unpack first and last of this subsequence
            state, prev_player, action, _, _, _ = seq[0]
            _, _, _, _, next_state, done = seq[-1]
            # normalize & push
            ret_norm.update(R_clipped) # Ensure R_clipped is used here if you want consistency
            R_norm = ret_norm.normalize(R_clipped) # Ensure R_clipped is used here if you want consistency
            memory.push(state, action, R_norm, next_state, done) 
        # finally clear the buffer
        nstep_buffer.buffer.clear()

        # Track episode results
        episode_rewards.append(total_reward)
        win_rates.append(1 if total_reward > 0 else 0)
        
        # # Update target network periodically
        # if episode % TARGET_UPDATE == 0:
        #     target_net.load_state_dict(policy_net.state_dict())
        
        # Log progress
        if episode % 10 == 0:
            # Calculate metrics
            avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards[-100:]), 100)
            avg_loss = sum(losses) / len(losses) if losses else 0
            win_rate = sum(win_rates) / len(win_rates) * 100 if win_rates else 0
            
            # Calculate elapsed time and estimated time remaining
            elapsed_time = time.time() - start_time
            time_per_episode = elapsed_time / (episode + 1)
            remaining_episodes = config.NUM_EPISODES - episode - 1
            remaining_time = datetime.timedelta(seconds=int(remaining_episodes * time_per_episode))
            
            # Print metrics
            print(f"Episode: {episode+1}/{config.NUM_EPISODES} | "
                  f"Mode: {'SelfPlay' if use_selfplay else f'BaseAgent({opponent.__name__ if opponent else None})'} | "
                  f"DQN as: {'White' if dqn_player == 1 else 'Black' if dqn_player == -1 else 'Both'} | "
                  f"Moves: {move_count} | "
                  f"Epsilon: {eps_threshold:.4f} | "
                  f"Reward: {total_reward:.2f} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Win Rate: {win_rate:.2f}% | "
                  f"Avg Loss: {avg_loss:.6f} | "
                  f"Memory Size: {len(memory)} | "
                  f"Remaining: {remaining_time}")
            
            # Save model checkpoint and evaluate
            if episode % 100 == 0:
                torch.save(policy_net.state_dict(), os.path.join("models", f"dqn_{episode}.pth"))

                # Evaluate against base agents
                if base_agents_list:
                    eval_results = evaluate_against_base_agents(policy_net, base_agents_list)
                    print("Evaluation against base agents:")
                    for agent_name, win_rate in eval_results.items():
                        print(f"  {agent_name}: {win_rate:.1%}")
                else:
                    print("Warning: Base agents list is empty. Skipping evaluation.")
    
    # Save final model
    torch.save(policy_net.state_dict(), os.path.join("models", "dqn_final.pth"))
    print("Training completed. Final model saved.")

if __name__ == "__main__":
    main()