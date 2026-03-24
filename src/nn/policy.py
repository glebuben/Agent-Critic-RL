import torch
from torch import nn

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_size=256):

        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(
            hidden_size, action_dim * 2
        )  # Output mean and log_std for each action dimension
        self.action_bound = action_bound

    def forward(self, states_batch):

        x = torch.relu(self.fc1(states_batch))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_actions(
        self,
        states_batch,
        reparameterize=True,
        return_log_prob=True,
        sample_deterministic=False,
    ):

        """
        Sample actions from the policy given a batch of states.
        Return actions batch and log probs 
        """

        x = self.forward(states_batch)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        std = torch.exp(log_std)
        normal_dist = torch.distributions.Normal(loc = mean, scale = std)

        if sample_deterministic:
            actions = mean  # Use mean for deterministic action selection
        else:
            if reparameterize:
                actions = normal_dist.rsample()  # Reparameterization trick
            else:
                actions = normal_dist.sample()
        squashed_actions = torch.tanh(actions)

        if return_log_prob and not sample_deterministic:

            # Sum log probabilities for all action dimensions
            normal_log_prob = normal_dist.log_prob(actions).sum(dim=-1)

            # Adjust log prob for tanh transformation, ignore scaling factor
            log_jacobian = torch.log(1 - squashed_actions.pow(2) + 1e-6).sum(dim=-1)
            log_prob = normal_log_prob - log_jacobian
        else:
            log_prob = None
        return squashed_actions * self.action_bound, log_prob

    @classmethod
    def load(cls, path):
        dict_loaded = torch.load(path)
        model = cls(
            state_dim=dict_loaded["state_dim"],
            action_dim=dict_loaded["action_dim"],
            action_bound=dict_loaded["action_bound"],
            hidden_size=dict_loaded["hidden_size"],
        )
        model.load_state_dict(dict_loaded["state_dict"])
        return model

    def save(self, path):
        dict_to_save = {
            "state_dict": self.state_dict(),
            "state_dim": self.fc1.in_features,
            "action_dim": self.fc3.out_features // 2,
            "action_bound": self.action_bound,
            "hidden_size": self.fc1.out_features,
        }
        torch.save(dict_to_save, path)
