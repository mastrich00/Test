import torch 
import hex_engine as engine


def select_action(root, temperature=1.0):
    if not root.children:
        raise ValueError("select_action: No children found in root.")

    visits = torch.tensor([child.visit_count for child in root.children.values()], dtype=torch.float32)
    actions = list(root.children.keys())

    if temperature == 0:
        # Greedy choice
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    # Schutz vor extrem kleiner Temperatur
    temperature = max(temperature, 1e-3)

    # Softmax-ähnliches Sampling
    probs = (visits ** (1.0 / temperature))
    probs = probs / probs.sum()

    action_idx = torch.multinomial(probs, 1).item()
    return actions[action_idx]



def copy_game(game):
    new_game = engine.hexPosition(size=game.size)
    new_game.board = [row[:] for row in game.board]
    new_game.player = game.player

    if hasattr(game, "winner"):
        new_game.winner = game.winner

    # Falls weitere relevante Felder existieren, ebenfalls kopieren
    for attr in ["move_history", "turn_count"]:
        if hasattr(game, attr):
            setattr(new_game, attr, getattr(game, attr))                                                                

    return new_game