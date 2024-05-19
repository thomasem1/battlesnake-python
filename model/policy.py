# Load up our dependencies
import os
import torch
import typing

from .utils import PredictionPolicy, SnakePolicyBase, GROUP_SHOUT

from gym import spaces

OBSERVATION_SPACE = (18, 23, 23)
ACTION_SPACE = spaces.Discrete(4)

def create_policy(obs_space, act_space, base):
    """ Returns a wrapped policy for use in the gym """
    return PredictionPolicy(obs_space, act_space, base=base)

def get_observation(game_state: typing.Dict):
    """
    layer0: snake health on heads {0,...,100}
    layer1: snake bodies {0,1}
    layer2: segment numbers {0,...,255}
    layer3: snake length >= player {0,1}
    layer4: food {0,1}
    layer5: gameboard {0,1}
    layer6: head_mask {0,1}
    layer7: double_tail_mask {0,1}
    layer8: snake bodies >= us {0,1}
    layer9: snake bodies < us {0,1}
    layer10-16: Alive count
    layer17: teammate snake bodies {0,1}
    """

    # Get game state info
    snakes = game_state['board']['snakes']
    food = game_state['board']['food']
    width = game_state['board']['width']
    height = game_state['board']['height']
    me = game_state['you']
    my_length = len(me['body'])

    # Initialize observation
    observation = torch.zeros(1, 18, 23, 23)

    # Layer 6: head_mask
    observation[0, 6, me['head']['x'], me['head']['y']] = 1

    alive_snakes = 0
    for snake in snakes:
        if snake['health'] <= 0:
            # Skip dead snakes
            continue
        alive_snakes += 1
        snake_length = len(snake['body'])

        # Layer 0: snake health on heads
        observation[0, 0, snake['head']['x'], snake['head']['y']] = snake['health']

        i = 0
        tail_1, tail_2 = None, None
        for body_part in reversed(snake['body']):
            if i == 0:
                tail_1 = body_part
            elif i == 1:
                tail_2 = body_part
                if tail_1 == tail_2:
                    # Layer 7: double_tail_mask
                    observation[0, 7, tail_1['x'], tail_1['y']] = 1
            
            # Layer 1: snake bodies
            observation[0, 1, body_part['x'], body_part['y']] = 1

            # Layer 2: segment numbers
            observation[0, 2, body_part['x'], body_part['y']] = min(i + 1, 255)
            i += 1

            # Layer 17: teammate snake bodies
            if snake['shout'] == GROUP_SHOUT:
                observation[0, 17, body_part['x'], body_part['y']] = 1
            
            if snake['id'] != me['id']:
                # Layer 8: snake bodies >= us
                if snake_length >= my_length:
                    observation[0, 8, body_part['x'], body_part['y']] = 1
                # Layer 9: snake bodies < us
                if snake_length < my_length:
                    observation[0, 9, body_part['x'], body_part['y']] = 1
        
        # Layer 3: snake length >= player
        if snake['id'] != me['id']:
            observation[0, 3, snake['head']['x'], snake['head']['y']] = 1 if snake_length >= my_length else 0
    
    # Layer 4: food
    for food_item in food:
        observation[0, 4, food_item['x'], food_item['y']] = 1
    
    for x in range(width):
        for y in range(height):
            # Layer 5: gameboard
            observation[0, 5, x, y] = 1

            # Layer 10-16: Alive count
            observation[0, 10 + alive_snakes, x, y] = 1
    return observation

class PolicyAgent:
    def __init__(self, device_type='cpu'):
        self.device = torch.device(device_type)        
        self.policy = create_policy(OBSERVATION_SPACE, ACTION_SPACE, SnakePolicyBase)
        self.policy.to(self.device)

    def load_policy(self, path):
        if os.path.exists(path):
            print("Loading policy from", path)
            self.policy.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print("File does not exist, using randomly initialized policy")

    def use_policy(self, obs):
        with torch.no_grad():
            inp = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action, _ = self.policy.predict(inp, deterministic=True, device=self.device)
        print("Action:", action)
        print("ACTION TYPE: ", type(action))
        return action
