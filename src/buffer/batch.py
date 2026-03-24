class ReplayBatch:
    """
    A batch of transitions sampled from the replay buffer, stored as PyTorch tensors.
    """
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones