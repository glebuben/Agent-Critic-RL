from src.sac.step import episode_step
from src.sac.utils import SACNetworks
from src.buffer import ReplayBuffer
from src.metrics import MetricsManager
import gymnasium as gym
import yaml
from tqdm.auto import tqdm
import os


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def generate_unique_ckpt_dir(base_dir):
    import os
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_dir = os.path.join(base_dir, str(timestamp))
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir


def train_sac(config_path):

    config = load_config(config_path)
    env = gym.make("HalfCheetah-v5")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = config.get("device", "cpu")

    ckpt_dir = config.get("ckpt_dir", None)
    if ckpt_dir is not None:
        ckpt_dir = generate_unique_ckpt_dir(ckpt_dir)
        print(f"Checkpoint directory set to: {ckpt_dir}")

    with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    networks = SACNetworks(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=config["action_bound"],
        hidden_size=config["hidden_size"],
        lr=config["lr"],
        number_of_qs=config["number_of_qs"],
        ckpt_dir=ckpt_dir,
    )

    networks.to_device(device)

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=config["buffer_size"],
        output_device=device,
    )

    env_steps = 0
    max_env_steps = config["max_env_steps"]
    best_episode_return = float("-inf")
    manager = MetricsManager()

    pbar = tqdm(total=max_env_steps, desc="Training SAC Agent", unit="env step")

    try:
        while env_steps < max_env_steps:
            steps, episode_return, total_distance = episode_step(
                env=env,
                networks=networks,
                replay_buffer=replay_buffer,
                batch_size=config["batch_size"],
                alpha=config["alpha"],
                tau=config["tau"],
                gamma=config["gamma"],
                gradient_steps=config["gradient_steps"],
            )
            env_steps += steps

            # track the best model
            if episode_return > best_episode_return:
                best_episode_return = episode_return
                networks.save()

            manager.update(env_steps, episode_return, total_distance)

            pbar.set_postfix(
                {
                    "Episode Return": f"{episode_return:.2f}",
                    "Total Distance": f"{total_distance:.2f}",
                }
            )
            pbar.update(steps)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        manager.save(os.path.join(ckpt_dir, "training_metrics.npz"))
    finally:
        pbar.close()
