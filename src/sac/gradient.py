import torch
import torch.nn.functional as F
from src.sac.utils import SACNetworks
from src.buffer import ReplayBatch

def gradient_step(
    networks: SACNetworks,
    replay_batch: ReplayBatch,
    alpha: float,
    tau: float,
    gamma: float,
    verbose: bool = False,
):

    states_batch = replay_batch.states
    actions_batch = replay_batch.actions
    rewards_batch = replay_batch.rewards
    next_states_batch = replay_batch.next_states
    dones_batch = replay_batch.dones

    # Compute loss for value network
    with torch.no_grad():
        policy_actions, log_prob = networks.policy.sample_actions(
            states_batch, reparameterize=False, return_log_prob=True
        )
        value_target = networks.get_qvalues(states_batch, policy_actions) - alpha*log_prob

    networks.value_optimizer.zero_grad()
    value_loss = F.mse_loss(networks.value_function(states_batch), value_target)
    value_loss.backward()
    networks.value_optimizer.step()

    # Compute loss for Q-function(s)
    for q_func, q_optimizer in zip(networks.q_function_list, networks.q_optimizers):
        with torch.no_grad():
            next_value = networks.target_value_function(next_states_batch)
            q_target = rewards_batch + gamma * (1 - dones_batch) * next_value

        q_loss = F.mse_loss(q_func(states_batch, actions_batch), q_target)
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

    # Compute loss for policy network
    policy_actions, log_prob = networks.policy.sample_actions(
        states_batch, reparameterize=True, return_log_prob=True
    )
    policy_loss = (alpha * log_prob - networks.get_qvalues(states_batch, policy_actions)).mean()

    networks.policy_optimizer.zero_grad()
    policy_loss.backward()
    networks.policy_optimizer.step()

    # Soft update target value network
    networks.target_value_function.smooth_update(networks.value_function, tau)

    if verbose:
        print('Value Loss:', value_loss.item())
        print('Q Loss:', q_loss.item())
        print('Policy Loss:', policy_loss.item())
        print("avg target Value:", next_value.mean().item())
        print("avg Q Value:", networks.get_qvalues(states_batch, policy_actions).mean().item())
        print("avg Value:", networks.value_function(states_batch).mean().item())
        print("avg Log Prob:", log_prob.mean().item())
        print("avg q_target:", q_target.mean().item())
        print("avg v_target:", value_target.mean().item())
        print("max Reward:", rewards_batch.max().item())
        print('---')