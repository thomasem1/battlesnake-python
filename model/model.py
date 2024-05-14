# Load up our dependencies
import numpy as np
import os
import time
import torch
import matplotlib.pyplot as plt

from gym_battlesnake.gymbattlesnake import BattlesnakeEnv
from a2c_ppo_acktr.algo import PPO
from a2c_ppo_acktr.storage import RolloutStorage
from sklearn.base import BaseEstimator

from tqdm.notebook import tqdm
from utils import PredictionPolicy, SnakePolicyBase

def create_policy(obs_space, act_space, base):
    """ Returns a wrapped policy for use in the gym """
    return PredictionPolicy(obs_space, act_space, base=base)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class RLAgent:
    def __init__(self, n_envs=50, n_steps=400, num_updates=50):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.num_updates = num_updates
        self.value_loss_coef = None
        self.entropy_coef = None
        self.max_grad_norm = None
        self.clip_param = None
        self.ppo_epoch = None
        self.num_mini_batch = None
        self.eps = None
        self.lr = None
        
        self.lengths = None
        self.rewards = None
        self.agent = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.reset_env()
    
    def reset_env(self):
        self.env = BattlesnakeEnv(n_threads=1, n_envs=self.n_envs)
        self.rollouts = RolloutStorage(self.n_steps,
                                    self.n_envs,
                                    self.env.observation_space.shape,
                                    self.env.action_space,
                                    self.n_steps)
        self.env.close()

        self.policy = create_policy(self.env.observation_space.shape, self.env.action_space, SnakePolicyBase)
        self.best_old_policy = create_policy(self.env.observation_space.shape, self.env.action_space, SnakePolicyBase)
        
        self.env = BattlesnakeEnv(n_threads=2, n_envs=self.n_envs, opponents=[self.policy for _ in range(2)], device=self.device, teammates=[self.policy])
        obs = self.env.reset()
        self.rollouts.obs[0].copy_(torch.tensor(obs))

    def set_agent(self, value_loss_coef, entropy_coef, max_grad_norm, clip_param, ppo_epoch, num_mini_batch, eps, lr):
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.eps = eps
        self.lr = lr

        self.agent = PPO(self.policy,
                        value_loss_coef=self.value_loss_coef,
                        entropy_coef=self.entropy_coef,
                        max_grad_norm=self.max_grad_norm,
                        clip_param=self.clip_param,
                        ppo_epoch=self.ppo_epoch,
                        num_mini_batch=self.num_mini_batch,
                        eps=self.eps,
                        lr=self.lr
                        )

    # Let's define a method to check our performance against an older policy
    # Determines an unbiased winrate check
    def check_performance(self, n_opponents=2, n_envs=1000, steps=1500):
        test_env = BattlesnakeEnv(n_threads=os.cpu_count(), n_envs=n_envs, opponents=[self.best_old_policy for _ in range(n_opponents)], device=self.device, teammates=[self.policy])
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
                inp = torch.tensor(obs, dtype=torch.float32).to(self.device)
                action, _ = self.policy.predict(inp, deterministic=True, device=self.device)
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

    def get_episode_length(self):
        return self.lengths[-1] if self.lengths else 0

    def train(self):
        # Send our network and storage to the gpu
        self.policy.to(self.device)
        self.best_old_policy.to(self.device)
        self.rollouts.to(self.device)

        # Record mean values to plot at the end
        rewards = []
        value_losses = []
        lengths = []

        start = time.time()
        for j in range(self.num_updates):
            episode_rewards = []
            episode_lengths = []
            # Set
            self.policy.eval()
            print(f"Iteration {j+1}: Generating rollouts")
            for step in tqdm(range(self.n_steps)):
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.policy.act(self.rollouts.obs[step],
                                                                    self.rollouts.recurrent_hidden_states[step],
                                                                    self.rollouts.masks[step])
                obs, reward, done, infos = self.env.step(action.cpu().squeeze())
                obs = torch.tensor(obs)
                reward = torch.tensor(reward).unsqueeze(1)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                        episode_lengths.append(info['episode']['l'])

                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                self.rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.policy.get_value(
                    self.rollouts.obs[-1],
                    self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]
                ).detach()
                
            # Set the policy to be in training mode (switches modules to training mode for things like batchnorm layers)
            self.policy.train()

            print("Training policy on rollouts...")
            # We're using a gamma = 0.99 and lambda = 0.95
            self.rollouts.compute_returns(next_value, True, 0.99, 0.95, False)
            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)
            self.rollouts.after_update()

            # Set the policy into eval mode (for batchnorms, etc)
            self.policy.eval()
            
            total_num_steps = (j + 1) * self.n_envs * self.n_steps
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
                winrate = self.check_performance()
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
                    self.best_old_policy.load_state_dict(self.policy.state_dict())
                else:
                    print("Policy has not learned enough yet... keep training!")
                print("-" * 80)

        self.lengths = lengths
        self.rewards = rewards

    def plot_results(self):
        plt.clf()
        plt.title("Average episode length")
        plt.ylabel("Length")
        plt.xlabel("Iteration")
        plt.plot(self.lengths)
        plt.show()

        plt.title("Average episode reward")
        plt.ylabel("Reward")
        plt.xlabel("Iteration")
        plt.plot(self.rewards)
        plt.show()

    def save_policy(self, path="policy.pth"):
        torch.save(self.policy.state_dict(), path)

class RLEstimator(BaseEstimator):
    def __init__(self, value_loss_coef=0.1, entropy_coef=0.001, max_grad_norm=0.1, clip_param=0.1,
                 ppo_epoch=3, num_mini_batch=16, eps=1e-6, lr=0.0001):
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.eps = eps
        self.lr = lr
        self.rl_agent = RLAgent(1, 400, 50)

    def fit(self, X, y=None):
        self.rl_agent.reset_env()
        self.rl_agent.set_agent(self.value_loss_coef, self.entropy_coef, self.max_grad_norm,
                                self.clip_param, self.ppo_epoch, self.num_mini_batch,
                                self.eps, self.lr)
        self.rl_agent.train()
        return self

    def score(self, X, y=None):
        # Evaluate the agent and return the negative of the episode length
        # (since GridSearchCV tries to maximize the score)
        return self.rl_agent.get_episode_length()
