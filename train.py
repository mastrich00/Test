import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from submission.model import HexNet
from submission.mcts import MCTS
import hex_engine as engine
import os
import matplotlib.pyplot as plt
from submission.utils import select_action
import submission.config as config
import pickle
import numpy as np
from submission import base_agents
from evaluation_positions import get_test_positions
from submission.utils import copy_game


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ReplayBuffer = deque(maxlen=50000)
BATCH_SIZE = 64 #32–256
NUM_SELFPLAY_GAMES = 5000
NUM_SIMULATIONS = 100 #50–800 (Train: 25–100, Eval: 400–800)
LEARNING_RATE = 1e-4
value_loss_coef = 0.25
entropy_coef = 0.005


net = HexNet(config.BOARD_SIZE).to(device)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

base_agents_list = [
    base_agents.random_agent,
    base_agents.greedy_center_agent,
    base_agents.edge_agent,
    base_agents.avoid_center_agent,
    base_agents.center_loving_agent,
    base_agents.blocking_agent,
    base_agents.greedy_connection_agent,
    base_agents.first_available_agent,
    base_agents.diagonal_agent,
    base_agents.mirror_agent
]

def choose_opponent(game_idx, num_games=NUM_SELFPLAY_GAMES):
    phase1_end = int(0.2 * num_games)  # 20%
    phase2_end = int(0.5 * num_games)  # 50%
    phase3_end = int(1.0 * num_games)  # 100%
    
    if game_idx <= phase1_end:
        baseline_prob = 0.5
    elif game_idx <= phase2_end:
        baseline_prob = 0.25
    elif game_idx <= phase3_end:
        baseline_prob = 0.1
    else:
        baseline_prob = 0.0

    if random.random() < baseline_prob:
        return random.choice(base_agents_list)
    else:
        return None  # Dein trainierter Agent

def smooth(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')


def evaluate_test_positions(net):
    positions = get_test_positions()
    net.eval()

    for idx, game in enumerate(positions):
        mcts = MCTS(game, net, n_simulations=100)  # kannst auch weniger nehmen
        root = mcts.run()

        # Zeige Top-Move & Value
        best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        policy_logits, value = net(board_to_tensor(game).unsqueeze(0).to(device))
        probs = torch.softmax(policy_logits, dim=1).detach().cpu().numpy().reshape(config.BOARD_SIZE, config.BOARD_SIZE)

        print(f"\nTeststellung {idx+1}:")
        print(f"Netz-Value: {value.item():.3f}")
        print(f"Empfohlener Zug: {game.scalar_to_coordinates(best_action)}")

        # Visualisierung der Policy
        import matplotlib.pyplot as plt
        plt.imshow(probs, cmap='viridis')
        plt.colorbar()
        plt.title(f"Policy Teststellung {idx+1}")
        plt.savefig(f"policy_test_{idx+1}.png")
        plt.close()


def board_to_tensor(game):
    board = game.board
    size = game.size
    player = game.player
    player_plane = [[1 if cell == player else 0 for cell in row] for row in board]
    opponent_plane = [[1 if cell == -player else 0 for cell in row] for row in board]
    state = [player_plane, opponent_plane]
    tensor = torch.tensor(state, dtype=torch.float32)
    return tensor

def get_temperature(game_idx, max_games, min_temp=0.1, max_temp=1.0):
    ratio = max(0, 1 - game_idx / max_games)
    return min_temp + (max_temp - min_temp) * ratio


def selfplay_game(game_idx):
    turns = 0
    game = engine.hexPosition(size=config.BOARD_SIZE)
    states = []
    mcts_probs = []
    current_players = []

    while game.winner == 0:
        mcts = MCTS(game, net, n_simulations=NUM_SIMULATIONS)
        root = mcts.run()

        temperature = get_temperature(game_idx, NUM_SELFPLAY_GAMES)
        action = select_action(root, temperature=temperature)

        # MCTS probabilities (visit counts)
        visit_counts = torch.zeros(config.BOARD_SIZE * config.BOARD_SIZE)
        total_visits = sum(child.visit_count for child in root.children.values())
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count / total_visits

        states.append(board_to_tensor(game))
        mcts_probs.append(visit_counts)
        current_players.append(game.player)

        # choose next move
        # action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        game.move(game.scalar_to_coordinates(action))

        turns += 1

    # winner (1, -1)
    z = game.winner

    # values for all states in the game
    results = []
    max_turns = config.BOARD_SIZE * config.BOARD_SIZE
    for player in current_players:
        reward = 1 if z == player else -1
        penalty = 0.01 * (turns / max_turns)  # skaliert zwischen 0 und 0.01
        reward -= penalty  # je länger das Spiel, desto geringer der Reward
        results.append(reward)

    return list(zip(states, mcts_probs, results))

def selfplay_game_with_baselines(game_idx):
    turns = 0
    game = engine.hexPosition(size=config.BOARD_SIZE)
    states, mcts_probs, current_players = [], [], []

    opponent = choose_opponent(game_idx)

    while game.winner == 0:
        if opponent is None:
            # MCTS Agent (dein Netz)
            mcts = MCTS(game, net, n_simulations=NUM_SIMULATIONS)
            root = mcts.run()
            temperature = get_temperature(game_idx, NUM_SELFPLAY_GAMES)
            action = select_action(root, temperature=temperature)

            visit_counts = torch.zeros(config.BOARD_SIZE * config.BOARD_SIZE)
            total_visits = sum(child.visit_count for child in root.children.values())
            for act, child in root.children.items():
                visit_counts[act] = child.visit_count / total_visits

        else:
            # Baseline Agent Zug
            action_set = game.get_action_space()
            action_coords = opponent(game.board, action_set)
            action = game.coordinate_to_scalar(action_coords)

            # Für Baseline Aktionen kannst du 'uniform' probs annehmen oder zeros:
            visit_counts = torch.zeros(config.BOARD_SIZE * config.BOARD_SIZE)

        states.append(board_to_tensor(game))
        mcts_probs.append(visit_counts)
        current_players.append(game.player)

        game.move(game.scalar_to_coordinates(action))
        turns += 1

    # Werteberechnung bleibt gleich
    z = game.winner
    results = []
    max_turns = config.BOARD_SIZE * config.BOARD_SIZE
    for player in current_players:
        reward = 1 if z == player else -1
        penalty = 0.01 * (turns / max_turns)
        reward -= penalty
        results.append(reward)

    return list(zip(states, mcts_probs, results))

def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()


def train():
    if len(ReplayBuffer) < BATCH_SIZE:
        return None

    batch = random.sample(ReplayBuffer, BATCH_SIZE)
    states, mcts_probs, results = zip(*batch)

    states = torch.stack(states).to(device)  # (B, 2, size, size)
    mcts_probs = torch.stack(mcts_probs).to(device)  # (B, size*size)
    results = torch.tensor(results, dtype=torch.float32).to(device)  # (B,)

    optimizer.zero_grad()
    policy_logits, values = net(states)
    values = values.view(-1)

    # Policy loss
    policy_log_probs = torch.log_softmax(policy_logits, dim=1)
    policy_loss = -(mcts_probs * policy_log_probs).sum(dim=1).mean()
    
    # Entropy
    policy_probs = torch.softmax(policy_logits, dim=1)
    entropy = compute_entropy(policy_probs)

    # Value loss
    value_loss = loss_fn(values, results)

    #loss = policy_loss + value_loss
    loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
    loss.backward()
    optimizer.step()

    return loss.item(), entropy

def evaluate_agents(agent1, agent2, num_games=20):
    agent1.eval()
    agent2.eval()
    wins_agent1 = 0
    wins_agent2 = 0

    for i in range(num_games):
        game = engine.hexPosition(size=config.BOARD_SIZE)
        players = [agent1, agent2] if i % 2 == 0 else [agent2, agent1]  # abwechselnd anfangen

        while game.winner == 0:
            net = players[0] if game.player == 1 else players[1]
            mcts = MCTS(copy_game(game), net, n_simulations=NUM_SIMULATIONS)
            root = mcts.run()
            action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
            game.move(game.scalar_to_coordinates(action))

        if game.winner == 1:
            if players[0] == agent1:
                wins_agent1 += 1
            else:
                wins_agent2 += 1
        elif game.winner == -1:
            if players[1] == agent1:
                wins_agent1 += 1
            else:
                wins_agent2 += 1
    
    total = wins_agent1 + wins_agent2 
    print(f"Evaluation Result over {total} games:")
    print(f"Agent 1 Wins: {wins_agent1}")
    print(f"Agent 2 Wins: {wins_agent2}")
    print(f"Win Rate Agent 1: {wins_agent1 / total:.2%}")
    print(f"Win Rate Agent 2: {wins_agent2 / total:.2%}")


def main():
    global ReplayBuffer

    os.makedirs("plots", exist_ok=True)

    old_net = HexNet(config.BOARD_SIZE).to(device)

    try:
        checkpoint = torch.load("models/checkpoint.pth", map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Checkpoint loaded.")
    except FileNotFoundError:
        print("No Checkpoint found. Start training from scratch.")

    try:
        with open("replay_buffer.pkl", "rb") as f:
            ReplayBuffer = pickle.load(f)
        print(f"ReplayBuffer loaded ({len(ReplayBuffer)} entries).")
    except FileNotFoundError:
        ReplayBuffer = deque(maxlen=30000)
        print("New ReplayBuffer initialized.")

    # Old net for evaluation
    old_net.load_state_dict(net.state_dict())

    losses = []
    entropies = []
    rewards = []
    avg_rewards = []
    value_errors = []
    policy_confidences = []

    for i in range(NUM_SELFPLAY_GAMES):
        data = selfplay_game(i)
        #data = selfplay_game_with_baselines(i)
        rewards.append(data[0][2])
        ReplayBuffer.extend(data)

        result = train()
        if result is not None:
            loss, entropy = result
            losses.append(loss)
            entropies.append(entropy)

         # Value Prediction Error (MSE gegen True Result)
        with torch.no_grad():
            states, mcts_probs, results = zip(*list(ReplayBuffer)[-BATCH_SIZE:])
            states = torch.stack(states).to(device)
            results = torch.tensor(results, dtype=torch.float32).to(device)
            _, predicted_values = net(states)
            value_error = ((predicted_values.view(-1) - results) ** 2).mean().item()
            value_errors.append(value_error)

        # Policy Confidence (max Wahrscheinlichkeiten)
        with torch.no_grad():
            policy_logits, _ = net(states)
            policy_probs = torch.softmax(policy_logits, dim=1)
            max_conf = policy_probs.max(dim=1).values.mean().item()
            policy_confidences.append(max_conf)

        if i % 100 == 0:
            print(f"Game {i} finished; size of ReplayBuffer: {len(ReplayBuffer)}")

            if losses:
                avg_loss = sum(losses[-100:]) / len(losses[-100:])
                avg_entropy = sum(entropies[-100:]) / len(entropies[-100:])
                print(f"Average Loss (last 100): {avg_loss:.4f}")
                print(f"Average Entropy (last 100): {avg_entropy:.4f}")

            avg_reward = sum(entry[2] for entry in ReplayBuffer) / len(ReplayBuffer)
            avg_rewards.append(avg_reward)
            print(f"Average Reward in ReplayBuffer: {avg_reward:.3f}")

            # Evaluation in try/except
            # Teil 2: Evaluation auf festen Teststellungen
            try:
                print("Evaluating fixed test positions...")
                evaluate_test_positions(net)
            except Exception as e:
                print(f"Evaluation of test positions failed: {e}")

            # Evaluation gegen vorheriges Netz
            try:
                print("Evaluating new agent against previous agent...")
                evaluate_agents(old_net, net)
            except Exception as e:
                print(f"Evaluation failed: {e}")
            
            plt.figure()
            plt.plot(avg_rewards)
            plt.title("Average Reward im ReplayBuffer")
            plt.xlabel("Evaluation Step")
            plt.ylabel("Average Reward")
            plt.savefig("plots/avg_reward.png")
            plt.close()

            plt.figure()
            plt.plot(value_errors)
            plt.title("Value Prediction MSE")
            plt.xlabel("Training Step")
            plt.ylabel("MSE")
            plt.savefig("plots/value_error.png")
            plt.close()

            plt.figure()
            plt.plot(policy_confidences)
            plt.title("Durchschnittliche Policy-Confidence (max Softmax)")
            plt.xlabel("Training Step")
            plt.ylabel("Avg Max Policy Prob")
            plt.savefig("plots/policy_confidence.png")
            plt.close()


            # Checkpoint und ReplayBuffer immer speichern
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, "models/checkpoint.pth")
            print("checkpoint.pth gespeichert.")

            with open("replay_buffer.pkl", "wb") as f:
                pickle.dump(ReplayBuffer, f)
            print("replay_buffer.pkl gespeichert.")

            # Backup current net as old_net
            old_net = HexNet(config.BOARD_SIZE).to(device)
            old_net.load_state_dict(net.state_dict())

    os.makedirs("models", exist_ok=True)  # create folder if not already existing

    # Save weights of model
    torch.save(net.state_dict(), "models/hexnet.pth")
    print("Model saved!")

    # Plot Loss
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.savefig("plots/training_loss.png")
    print("Saved training_loss.png")

    # Plot Entropy
    plt.figure()
    plt.plot(entropies)
    plt.title("Policy Entropy")
    plt.xlabel("Training Step")
    plt.ylabel("Entropy")
    plt.savefig("plots/policy_entropy.png")
    print("Saved policy_entropy.png")

    # Plot Rewards
    plt.figure()
    plt.plot(smooth(rewards))
    plt.title("Reward Verlauf (1 = Win, -1 = Loss)")
    plt.xlabel("Selfplay Game")
    plt.ylabel("Reward")
    plt.savefig("training_rewards.png")
    print("Saved training_rewards.png")

if __name__ == "__main__":
    main()
