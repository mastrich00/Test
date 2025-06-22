import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, board_size, action_size, hidden_size=128):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size
        
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * board_size * board_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers.
    Helps in training deeper networks by allowing gradients to flow better.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # The "skip connection"
        return F.relu(out)

class DuelingDQN(nn.Module):
    """
    A Dueling DQN with a ResNet-style convolutional backbone.
    """
    def __init__(self, board_size, action_size, num_channels=64, num_res_blocks=4):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        # Initial convolution to get into the feature space
        self.initial_conv = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False) # NOTE: in_channels is now 3
        self.initial_bn = nn.BatchNorm2d(num_channels)

        # A stack of residual blocks
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # --- Dueling Heads ---
        # Value Head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc = nn.Linear(32 * board_size * board_size, 1)

        # Advantage Head
        self.advantage_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.advantage_bn = nn.BatchNorm2d(32)
        self.advantage_fc = nn.Linear(32 * board_size * board_size, action_size)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Initial block
        x = F.relu(self.initial_bn(self.initial_conv(x)))

        # Residual tower
        x = self.residual_tower(x)

        # Value head
        val = F.relu(self.value_bn(self.value_conv(x)))
        val = val.view(val.size(0), -1)
        val = self.value_fc(val) # Shape: [batch, 1]

        # Advantage head
        adv = F.relu(self.advantage_bn(self.advantage_conv(x)))
        adv = adv.view(adv.size(0), -1)
        adv = self.advantage_fc(adv) # Shape: [batch, action_size]

        # Combine value and advantage to get Q-values
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values