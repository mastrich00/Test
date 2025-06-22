import random

def infer_player(board):
    our   = sum(cell == 1  for row in board for cell in row)
    their = sum(cell == -1 for row in board for cell in row)
    if our == their:
        return 1, -1    # White's turn
    else:
        return -1, 1    # Black's turn

# Random agent
def random_agent(board, action_set):
    return random.choice(action_set)

# Greedy-Center Agent
def greedy_center_agent(board, action_set):
    size = len(board)
    center = (size // 2, size // 2)
    return min(action_set, key=lambda move: (move[0]-center[0])**2 + (move[1]-center[1])**2)

# Edge Agent: picks a random edge move if available
def edge_agent(board, action_set):
    size = len(board)
    edge_moves = [mv for mv in action_set if mv[0] == 0 or mv[1] == 0 or mv[0] == size-1 or mv[1] == size-1]
    return random.choice(edge_moves) if edge_moves else random.choice(action_set)

# Avoid-Center Agent
def avoid_center_agent(board, action_set):
    size = len(board)
    center = (size // 2, size // 2)
    return max(action_set, key=lambda mv: (mv[0]-center[0])**2 + (mv[1]-center[1])**2)

# Center-Loving Agent
def center_loving_agent(board, action_set):
    size = len(board)
    center = (size // 2, size // 2)
    return min(action_set, key=lambda mv: (mv[0]-center[0])**2 + (mv[1]-center[1])**2)

# Greedy-Connection Agent: connects to own stones
# Uses infer_player to determine which stones to count

def greedy_connection_agent(board, action_set):
    size = len(board)
    player, _ = infer_player(board)

    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,-1)]
    def count_neighbors(mv):
        r, c = mv
        return sum(1 for dx,dy in neighbors
                   if 0 <= r+dx < size and 0 <= c+dy < size and board[r+dx][c+dy] == player)

    return max(action_set, key=count_neighbors)

# First-Available Agent: picks the first in action_set
def first_available_agent(board, action_set):
    return action_set[0]

# Blocking Agent: blocks opponent connections
def blocking_agent(board, action_set):
    size = len(board)
    _, opponent = infer_player(board)
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,-1)]
    threats = []
    for mv in action_set:
        r,c = mv
        if any(0 <= r+dx < size and 0 <= c+dy < size and board[r+dx][c+dy] == opponent
               for dx,dy in neighbors):
            threats.append(mv)
    return random.choice(threats) if threats else random.choice(action_set)

# Diagonal Agent: prefers any diagonal move at random
def diagonal_agent(board, action_set):
    diag = [mv for mv in action_set if mv[0] == mv[1]]
    return random.choice(diag) if diag else random.choice(action_set)

# Mirror Agent: mirrors the opponent's last move if possible
def mirror_agent(board, action_set):
    size = len(board)
    player, opponent = infer_player(board)

    # collect moves in play order; assume scan row-major approximates last
    opp_moves = [(r,c) for r in range(size) for c in range(size) if board[r][c] == opponent]
    if not opp_moves:
        return random.choice(action_set)
    last = opp_moves[-1]
    mirror = (size-1-last[0], size-1-last[1])
    return mirror if mirror in action_set else random.choice(action_set)
