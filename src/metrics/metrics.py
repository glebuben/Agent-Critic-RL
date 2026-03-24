import numpy as np


class MetricsManager:
    """Class for tracking episode return and total distance during training."""

    def __init__(self):

        self.reset()

    def update(self, step: int, reward: float, distance: float):
        self.n_steps.append(step)
        self.episode_returns.append(reward)
        self.total_distances.append(distance)

    def reset(self):
        self.n_steps = []
        self.episode_returns = []
        self.total_distances = []

    def save(self, filepath: str):

        np.savez(
            filepath,
            n_steps=np.array(self.n_steps),
            episode_returns=np.array(self.episode_returns),
            total_distances=np.array(self.total_distances),
        )
