# Load up our dependencies
import numpy as np
import os
import time
import torch
import matplotlib.pyplot as plt

from gym_battlesnake.gymbattlesnake import BattlesnakeEnv
from a2c_ppo_acktr.algo import PPO
from a2c_ppo_acktr.storage import RolloutStorage

from tqdm.notebook import tqdm
from utils import PredictionPolicy, SnakePolicyBase

def create_policy(obs_space, act_space, base):
    """ Returns a wrapped policy for use in the gym """
    return PredictionPolicy(obs_space, act_space, base=base)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Let's define a method to check our performance against an older policy
# Determines an unbiased winrate check
def check_performance(current_policy, opponent, n_opponents=3, n_envs=1000, steps=1500, device=torch.device('cpu')):
    test_env = BattlesnakeEnv(n_threads=os.cpu_count(), n_envs=n_envs, opponents=[opponent for _ in range(n_opponents)], device=device)
    obs = test_env.reset()
    wins = 0
    losses = 0
    completed = np.zeros(n_envs)
    count = 0
    lengths = []
    with torch.no_grad():
        # Simulate to a maximum steps across our environments, only recording the first result in each env.
        print("Running performance check")
        for step in tqdm(range(steps)):
            if count == n_envs:
                # Quick break
                print("Check Performance done @ step", step)
                break
            inp = torch.tensor(obs, dtype=torch.float32).to(device)
            action, _ = current_policy.predict(inp, deterministic=True, device=device)
            obs, reward, done, info = test_env.step(action.cpu().numpy().flatten())
            for i in range(test_env.n_envs):
                if completed[i] == 1:
                    continue # Only count each environment once
                if 'episode' in info[i]:
                    if info[i]['episode']['r'] == 1:
                        completed[i] = 1
                        count += 1
                        wins += 1
                        lengths.append(info[i]['episode']['l'])
                    elif info[i]['episode']['r'] == -1:
                        completed[i] = 1
                        losses += 1
                        count += 1
                        lengths.append(info[i]['episode']['l'])

    winrate = wins / n_envs
    print("Wins", wins)
    print("Losses", losses)
    print("Average episode length:", np.mean(lengths))
    return winrate

def train(policy, best_old_policy, agent, rollouts, env, device, num_updates, n_envs, n_steps, plot=False):
    # Send our network and storage to the gpu
    policy.to(device)
    best_old_policy.to(device)
    rollouts.to(device)

    # Record mean values to plot at the end
    rewards = []
    value_losses = []
    lengths = []

    start = time.time()
    for j in range(num_updates):
        episode_rewards = []
        episode_lengths = []
        # Set
        policy.eval()
        print(f"Iteration {j+1}: Generating rollouts")
        for step in tqdm(range(n_steps)):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = policy.act(rollouts.obs[step],
                                                                rollouts.recurrent_hidden_states[step],
                                                                rollouts.masks[step])
            obs, reward, done, infos = env.step(action.cpu().squeeze())
            obs = torch.tensor(obs)
            reward = torch.tensor(reward).unsqueeze(1)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_lengths.append(info['episode']['l'])

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = policy.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]
            ).detach()
            
        # Set the policy to be in training mode (switches modules to training mode for things like batchnorm layers)
        policy.train()

        print("Training policy on rollouts...")
        # We're using a gamma = 0.99 and lambda = 0.95
        rollouts.compute_returns(next_value, True, 0.99, 0.95, False)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        # Set the policy into eval mode (for batchnorms, etc)
        policy.eval()
        
        total_num_steps = (j + 1) * n_envs * n_steps
        end = time.time()
        
        lengths.append(np.mean(episode_lengths))
        rewards.append(np.mean(episode_rewards))
        value_losses.append(value_loss)
        
        # Every 5 iterations, we'll print out the episode metrics
        if (j+1) % 5 == 0:
            print("\n")
            print("=" * 80)
            print("Iteration", j+1, "Results")
            # Check the performance of the current policy against the prior best
            winrate = check_performance(policy, best_old_policy, device=device)
            print(f"Winrate vs prior best: {winrate*100:.2f}%")
            print(f"Median Length: {np.median(episode_lengths)}")
            print(f"Max Length: {np.max(episode_lengths)}")
            print(f"Min Length: {np.min(episode_lengths)}")

            # If our policy wins more than 30% of the games against the prior
            # best opponent, update the prior best.
            # Expected outcome for equal strength players is 25% winrate in a 4 player
            # match.
            if winrate > 0.3:
                print("Policy winrate is > 30%. Updating prior best model")
                best_old_policy.load_state_dict(policy.state_dict())
            else:
                print("Policy has not learned enough yet... keep training!")
            print("-" * 80)
        if plot:
            plot_results(lengths, rewards)

def plot_results(lengths, rewards):
    plt.clf()
    plt.title("Average episode length")
    plt.ylabel("Length")
    plt.xlabel("Iteration")
    plt.plot(lengths)
    plt.show()

    plt.title("Average episode reward")
    plt.ylabel("Reward")
    plt.xlabel("Iteration")
    plt.plot(rewards)
    plt.show()

def save_model(policy, path):
    torch.save(policy.state_dict(), path)

def train_model(n_envs, n_steps, num_updates, value_loss_coef, entropy_coef, max_grad_norm, clip_param, ppo_epoch, num_mini_batch, eps, lr):
    my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = BattlesnakeEnv(n_threads=1, n_envs=n_envs)
    rollouts = RolloutStorage(n_steps,
                            n_envs,
                            env.observation_space.shape,
                            env.action_space,
                            n_steps)
    env.close()

    policy = create_policy(env.observation_space.shape, env.action_space, SnakePolicyBase)
    best_old_policy = create_policy(env.observation_space.shape, env.action_space, SnakePolicyBase)
    agent = PPO(policy,
                value_loss_coef=value_loss_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                clip_param=clip_param,
                ppo_epoch=ppo_epoch,
                num_mini_batch=num_mini_batch,
                eps=eps,
                lr=lr)
    env = BattlesnakeEnv(n_threads=2, n_envs=n_envs, opponents=[policy for _ in range(3)], device=my_device)
    obs = env.reset()
    rollouts.obs[0].copy_(torch.tensor(obs))

    train(policy, best_old_policy, agent, rollouts, env, my_device, num_updates, n_envs, n_steps, plot=True)
    save_model(policy, "model.pth")
