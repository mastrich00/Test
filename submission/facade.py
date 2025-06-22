#trivial solution
#def agent (board, action_set):
#    return action_set[0]

#Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
#Please make sure that the agent does actually work with the provided Hex module.

import torch
from submission.model import HexNet
from submission.mcts import MCTS
import hex_engine as engine
from submission.utils import select_action
import config


_net = None

def load_model():
    global _net
    if _net is None:
        _net = HexNet(config.BOARD_SIZE)
        _net.load_state_dict(torch.load("models/hexnet.pth", map_location="cpu"))
        _net.eval()
    return _net

def detect_player(board):
    count = sum(cell != 0 for row in board for cell in row)
    return 1 if count % 2 == 0 else -1


def agent(board, action_set):
    """
    board: 2D list of ints, current board state
    action_set: list of valid moves (tuples)
    returns: chosen action (i, j)
    """
    if not action_set:
        raise ValueError("No valid actions available.")
    
    game = engine.hexPosition(size=config.BOARD_SIZE)
    game.board = [row[:] for row in board]
    game.player = detect_player(board)

    net = load_model()
    mcts = MCTS(game, net, n_simulations=getattr(config, "SIMULATIONS", 50))
    root = mcts.run()

    if not root.children:
        return action_set[0]  # fallback fallback
    
    # Pick action with highest visit count
    #best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
    action = select_action(root, temperature=0.1)
    return game.scalar_to_coordinates(action)
