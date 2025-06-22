import math
import numpy as np
import torch.nn.functional as F
import hex_engine as engine
import torch

class Node:
    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.children = {}  # action -> Node
        self.visit_count = 0
        self.value_sum = 0
        self.prior_prob = prior_prob

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, game, net, c_puct=1.0, n_simulations=100, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.game = game
        self.net = net
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = next(net.parameters()).device

    def run(self):
        root = Node(parent=None, prior_prob=1)
        for _ in range(self.n_simulations):
            game_copy = self.copy_game(self.game)
            self.simulate(game_copy, root)
        return root


    def simulate(self, game, node):
        if game.winner != 0:
            # Terminal node
            return -game.winner  # Perspective of current player

        if not node.children:
            # Leaf node expansion
            state_tensor = self.game_to_tensor(game)
            policy_logits, value = self.net(state_tensor)
            policy = F.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]

            valid_moves = game.get_action_space()
            for action in valid_moves:
                try: 
                    scalar_action = game.coordinate_to_scalar(action)
                    node.children[scalar_action] = Node(node, prior_prob=policy[scalar_action])
                except (IndexError, ValueError, KeyError) as e:
                    print(f"Warnung: Fehler bei Umwandlung von {action} -> scalar: {e}")
            if node.parent is None:
                self.add_dirichlet_noise(node)
            return -value.item()

        # Select move with max UCB score
        best_score = -float('inf')
        best_action = None
        for action, child in node.children.items():
            # UCB score balances exploitation (value) and exploration (prior_prob)
            u = child.value() + self.c_puct * child.prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
            if u > best_score:
                best_score = u
                best_action = action

        next_node = node.children[best_action]
        # Play action in game copy
        game.move(game.scalar_to_coordinates(best_action))

        v = self.simulate(game, next_node)

        next_node.value_sum += v
        next_node.visit_count += 1
        node.visit_count += 1

        return -v


    def game_to_tensor(self, game):
        # Convert the board to a tensor with shape (1, 2, size, size)
        board = game.board
        size = game.size
        player = game.player
        player_plane = [[1 if cell == player else 0 for cell in row] for row in board]
        opponent_plane = [[1 if cell == -player else 0 for cell in row] for row in board]
        state = [player_plane, opponent_plane]
        tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        tensor = tensor.to(next(self.net.parameters()).device)
        return tensor


    def copy_game(self, game):
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
    

    def add_dirichlet_noise(self, node):
        actions = list(node.children.keys())
        if not actions:
            return  # keine Aktionen, also kein Rauschen
        
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for i, action in enumerate(actions):
            node.children[action].prior_prob = \
                (1 - self.dirichlet_epsilon) * node.children[action].prior_prob + self.dirichlet_epsilon * noise[i]

