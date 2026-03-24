import torch
from torch import nn


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super(ValueNetwork, self).__init__()
        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, states_batch):
        x = torch.relu(self.fc1(states_batch))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze(-1)  # Return as (batch_size,) instead of (batch_size, 1)

    def smooth_update(self, online_network, tau):
        for target_param, online_param in zip(self.parameters(), online_network.parameters()):
            # in-place update to target network parameters
            target_param.data.copy_((1 - tau) * target_param.data + tau * online_param.data)

    @classmethod
    def load(cls, path, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dict_loaded = torch.load(path, map_location=device)
        model = cls(
            state_dim=dict_loaded["state_dim"],
            hidden_size=dict_loaded["hidden_size"],
        )
        model.load_state_dict(dict_loaded["state_dict"])
        return model.to(device)

    def save(self, path):
        dict_to_save = {
            "state_dict": self.state_dict(),
            "state_dim": self.state_dim,
            "hidden_size": self.fc1.out_features,
        }
        torch.save(dict_to_save, path)
