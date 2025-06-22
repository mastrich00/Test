import importlib.util
from submission.mcts import MCTS
import hex_engine as engine
from submission.model import HexNet
import submission.config as config
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
def load_trained_model(path="models/hexnet.pth"):
    model = HexNet(config.BOARD_SIZE).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# Wrapper for agent
def model_agent(model):
    def agent(board, action_set):
        game = engine.hexPosition(size=config.BOARD_SIZE)
        game.board = [row[:] for row in board]  # copy
        game.player = 1 if sum(cell != 0 for row in board for cell in row) % 2 == 0 else -1
        mcts = MCTS(game, model, n_simulations=100)
        root = mcts.run()
        action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return game.scalar_to_coordinates(action)
    return agent

# play the game
def play_game(agent1, agent2):
    game = engine.hexPosition(size=config.BOARD_SIZE)
    agents = {1: agent1, -1: agent2}
    while game.winner == 0:
        board = game.board
        action_set = game.get_action_space()
        move = agents[game.player](board, action_set)
        game.move(move)
    return game.winner

# Load base agents
def load_base_agents(path="base_agents.py"):
    spec = importlib.util.spec_from_file_location("base_agents", path)
    base_agents = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_agents)
    return base_agents

# Evaluate
def evaluate_against_base_agent(net, base_agent_fn, num_games=20):
    net.eval()
    wins, losses = 0, 0

    for i in range(num_games):
        game = engine.hexPosition(size=config.BOARD_SIZE)
        starts_first = (i % 2 == 0)

        while game.winner == 0:
            if (game.player == 1 and starts_first) or (game.player == -1 and not starts_first):
                mcts = MCTS(game, net, n_simulations=100)
                root = mcts.run()
                action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
                move = game.scalar_to_coordinates(action)
            else:
                move = base_agent_fn(game.board, game.get_action_space())

            game.move(move)

        if (game.winner == 1 and starts_first) or (game.winner == -1 and not starts_first):
            wins += 1
        elif (game.winner == -1 and starts_first) or (game.winner == 1 and not starts_first):
            losses += 1

    print(f"\nResults against {base_agent_fn.__name__}:")
    print(f"Wins:   {wins}")
    print(f"Losses: {losses}")
    print(f"Winrate: {wins / num_games:.2%}")

    return wins, losses

def main():
    # Load model
    net = load_trained_model("models/hexnet.pth")

    # Import base agents
    base_agents = load_base_agents("submission/base_agents.py")

    total_wins = 0
    total_losses = 0
    total_games = 0

    agent_list = [
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
    
    for agent_fn in agent_list:
        wins, losses = evaluate_against_base_agent(net, agent_fn, num_games=20)
        total_wins += wins
        total_losses += losses
        total_games += wins + losses

    print("\n==== Totals ====")
    print(f"Total Wins:   {total_wins}")
    print(f"Total Losses: {total_losses}")
    print(f"Total Winrate: {total_wins / total_games:.2%}")

if __name__ == "__main__":
    main()