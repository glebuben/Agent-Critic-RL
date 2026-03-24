import torch
import gymnasium as gym
from src.sac.utils import SACNetworks
from src.buffer import ReplayBuffer
from src.sac.gradient import gradient_step
from src.eval import evaluate_episode


def episode_step(
    env: gym.Env,
    networks: SACNetworks,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    alpha: float,
    tau: float,
    gamma: float,
    gradient_steps: int = 1,
):
    env_steps = 0
    device = replay_buffer.output_device

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(device)
    done = False

    while not done:

        action, _ = networks.policy.sample_actions(
            state.unsqueeze(0), reparameterize=False, return_log_prob=False
        )
        action = action.squeeze(0)
        next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = terminated or truncated

        replay_buffer.add(
            state.cpu().numpy(), action.cpu().numpy(), reward, next_state, terminated
        )

        state = torch.tensor(next_state, dtype=torch.float32).to(device)
        env_steps += 1

        if replay_buffer.size >= batch_size:
            for _ in range(gradient_steps):
                replay_batch = replay_buffer.sample(batch_size)
                gradient_step(
                    networks=networks,
                    replay_batch=replay_batch,
                    alpha=alpha,
                    tau=tau,
                    gamma=gamma,
                    verbose=False,
                )

    episode_return, total_distance = evaluate_episode(env, networks.policy, device)

    return env_steps, episode_return, total_distance
