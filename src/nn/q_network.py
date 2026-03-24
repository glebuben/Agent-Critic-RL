import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, states_batch, actions_batch):
        x = torch.cat([states_batch, actions_batch], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value.squeeze(-1)  # Return as (batch_size,) instead of (batch_size, 1)

    @classmethod
    def load(cls, path, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dict_loaded = torch.load(path, map_location=device)
        model = cls(
            state_dim=dict_loaded["state_dim"],
            action_dim=dict_loaded["action_dim"],
            hidden_size=dict_loaded["hidden_size"],
        )
        model.load_state_dict(dict_loaded["state_dict"])
        return model.to(device)

    def save(self, path):
        dict_to_save = {
            "state_dict": self.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": self.fc1.out_features,
        }
        torch.save(dict_to_save, path)
