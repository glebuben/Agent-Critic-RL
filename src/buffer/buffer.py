import torch
from src.buffer.batch import ReplayBatch


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transitions based on rolling arrays.
    """

    def __init__(self, state_dim, action_dim, max_size=int(1e6), output_device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.output_device = output_device

        self.state_buffer = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.action_buffer = torch.zeros((max_size, action_dim), dtype=torch.float32)
        self.reward_buffer = torch.zeros((max_size,), dtype=torch.float32)
        self.next_state_buffer = torch.zeros((max_size, state_dim), dtype=torch.float32)
        self.done_buffer = torch.zeros((max_size,), dtype=torch.int32)

    def add(self, state, action, reward, next_state, done):
        self.state_buffer[self.ptr] = torch.tensor(state, dtype=torch.float32)
        self.action_buffer[self.ptr] = torch.tensor(action, dtype=torch.float32)
        self.reward_buffer[self.ptr] = torch.tensor(reward, dtype=torch.float32)
        self.next_state_buffer[self.ptr] = torch.tensor(next_state, dtype=torch.float32)
        self.done_buffer[self.ptr] = torch.tensor(done, dtype=torch.int32)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,))
        return ReplayBatch(
            self.state_buffer[indices].to(self.output_device),
            self.action_buffer[indices].to(self.output_device),
            self.reward_buffer[indices].to(self.output_device),
            self.next_state_buffer[indices].to(self.output_device),
            self.done_buffer[indices].to(self.output_device),
        )
