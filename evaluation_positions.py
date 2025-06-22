import hex_engine as engine
import submission.config as config
from submission.utils import copy_game

def get_test_positions():
    size = config.BOARD_SIZE
    positions = []

    # 1. Leeres Board
    game = engine.hexPosition(size)
    positions.append(copy_game(game))

    # 2. Ein Zug in der Mitte
    game = engine.hexPosition(size)
    game.move((size // 2, size // 2))
    positions.append(copy_game(game))

    # 3. Zwei Züge (abwechselnd)
    game = engine.hexPosition(size)
    game.move((size // 2, size // 2))
    game.move((0, 0))
    positions.append(copy_game(game))

    return positions
