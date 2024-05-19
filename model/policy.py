# Load up our dependencies
import os
import torch

from .utils import PredictionPolicy, SnakePolicyBase

from gym import spaces

OBSERVATION_SPACE = (18, 23, 23)
ACTION_SPACE = spaces.Discrete(4)

def create_policy(obs_space, act_space, base):
    """ Returns a wrapped policy for use in the gym """
    return PredictionPolicy(obs_space, act_space, base=base)

class PolicyAgent:
    def __init__(self, device_type='cpu'):
        self.device = torch.device(device_type)        
        self.policy = create_policy(OBSERVATION_SPACE, ACTION_SPACE, SnakePolicyBase)
        self.policy.to(self.device)

    def load_policy(self, path):
        if os.path.exists(path):
            print("Loading policy from", path)
            self.policy.load_state_dict(torch.load(path))
        else:
            print("File does not exist, using randomly initialized policy")

    def use_policy(self, obs):
        with torch.no_grad():
            inp = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action, _ = self.policy.predict(inp, deterministic=True, device=self.device)
        print("Action:", action)
        print("ACTION TYPE: ", type(action))
        return action
