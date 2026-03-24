import torch

def evaluate_episode(env, policy, device):

    with torch.no_grad():
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        done = False

        total_distance = 0.0
        episode_return = 0.0

        while not done:
            action, _ = policy.sample_actions(
                state.unsqueeze(0),
                reparameterize=False,
                return_log_prob=False,
                sample_deterministic=True,
            )
            action = action.squeeze(0)
            next_state, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated or truncated

            episode_return += reward
            total_distance = info.get("x_position")

            state = torch.tensor(next_state, dtype=torch.float32).to(device)

    return episode_return, total_distance
