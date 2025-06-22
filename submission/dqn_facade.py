import torch
from submission.dqn_model import DuelingDQN as DQN
import submission.dqn_config as config
import hex_engine as engine
from submission.dqn_utils import board_to_tensor

_model = None

def load_model():
    global _model
    if _model is None:
        board_size = config.BOARD_SIZE
        action_size = board_size * board_size
        _model = DQN(board_size, action_size)
        _model.load_state_dict(torch.load("dqn_final.pth", map_location="cpu"))
        _model.eval()
    return _model

def agent(board, action_set):
    game = engine.hexPosition(size=config.BOARD_SIZE)
    game.board = [row[:] for row in board]
    
    model = load_model()
    state_tensor = board_to_tensor(game).unsqueeze(0)
    
    with torch.no_grad():
        q_values = model(state_tensor)
    
    valid_actions = []
    for action in action_set:
        scalar = game.coordinate_to_scalar(action)
        valid_actions.append((q_values[0, scalar].item(), action))
    
    return max(valid_actions, key=lambda x: x[0])[1]