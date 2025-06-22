import torch
import hex_engine as engine

def board_to_tensor(game):
    board = game.board
    
    # Channel 0: Current player's pieces (always from the perspective of the one to move)
    player_plane = [[1 if cell == game.player else 0 for cell in row] for row in board]
    
    # Channel 1: Opponent's pieces
    opponent_plane = [[1 if cell == -game.player else 0 for cell in row] for row in board]
    
    # Channel 2: A plane indicating whose turn it is (1 for White, -1 for Black)
    # This gives the network an absolute sense of color, which can be useful.
    turn_plane = [[game.player for _ in row] for row in board]
    
    state = [player_plane, opponent_plane, turn_plane]
    return torch.tensor(state, dtype=torch.float32)

def copy_game(game):
    new_game = engine.hexPosition(size=game.size)
    new_game.board = [row[:] for row in game.board]
    new_game.player = game.player
    new_game.winner = game.winner
    return new_game