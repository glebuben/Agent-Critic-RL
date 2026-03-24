import torch
from src.nn import Policy, QNetwork, ValueNetwork
from pathlib import Path

class SACNetworks:
    """
    A container for the policy, Q-function, and value function networks used in Soft Actor-Critic (SAC),
    as well as their optimizers.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        action_bound,
        hidden_size=256,
        lr=3e-4,
        number_of_qs=2,
        ckpt_dir=None
    ):

        warn_msg = "SAC typically uses either 1 or 2 Q-networks. Using more than 2 is uncommon and may lead to increased computational cost without significant benefits."

        assert number_of_qs in [1, 2], warn_msg

        self.policy = Policy(state_dim, action_dim, action_bound, hidden_size)
        
        # 1 or 2 Q-networks, for simple or double Q-learning respectively
        self.q_function_list = [
            QNetwork(state_dim, action_dim, hidden_size) for _ in range(number_of_qs)
        ]
        self.value_function = ValueNetwork(state_dim, hidden_size)

        # copy weights to target value function
        self.target_value_function = ValueNetwork(state_dim, hidden_size)
        self.target_value_function.load_state_dict(self.value_function.state_dict())

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizers = [
            torch.optim.Adam(q.parameters(), lr=lr) for q in self.q_function_list
        ]
        self.value_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=lr)

        self.ckpt_dir = ckpt_dir

    def to_device(self, device):
        self.policy.to(device)
        for q in self.q_function_list:
            q.to(device)
        self.value_function.to(device)
        self.target_value_function.to(device)

    def get_qvalues(self, states, actions):

        q_values = [qfunc(states, actions) for qfunc in self.q_function_list]
        return torch.min(torch.stack(q_values, dim=0), dim=0)[0]  # Take min across Q-functions

    def save(self):

        if self.ckpt_dir is None:
            raise ValueError("Checkpoint directory (ckpt_dir) is not set. Cannot save networks.")
        else:
            ckpt_dir = Path(self.ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.value_function.save(ckpt_dir / "value_function.pth")
            self.target_value_function.save(ckpt_dir / "target_value_function.pth")
            self.policy.save(ckpt_dir / "policy.pth")

            for idx, qfunc in enumerate(self.q_function_list):
                qfunc.save(ckpt_dir / f"q_function_{idx}.pth")