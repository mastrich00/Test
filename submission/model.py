import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_prob=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.05)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.05)
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)
    

class HexNet(nn.Module):
    def __init__(self, board_size, num_blocks=5):
        super().__init__()
        self.board_size = board_size

        # Input
        self.input_conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.05),
            nn.ReLU()
        )

        # Residual
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])

        # Policy Head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2, momentum=0.05),
            nn.ReLU()
        )
        self.fc_policy = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value Head
        self.value_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1, momentum=0.05),
            # nn.ReLU() # besser entfernen, damit Werte nicht unnötig begrenzt werden
        )
        self.fc_value1 = nn.Linear(board_size * board_size, 64)
        self.fc_value2 = nn.Linear(64, 1)

        self.apply(self._init_weights)

    # He-Initialization
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, 2, board_size, board_size)
        x = self.input_conv(x)
        x = self.res_blocks(x)

        # Policy Path
        p = self.policy_conv(x)
        p = p.flatten(start_dim=1)
        policy_logits = self.fc_policy(p)

        # Value Path
        v = self.value_conv(x)
        v = v.flatten(start_dim=1)
        v = F.relu(self.fc_value1(v))
        value = torch.tanh(self.fc_value2(v))

        return policy_logits, value
