import random
import typing
import torch

from model.policy import PolicyAgent

GROUP_SHOUT = "GROUP 4:20"

rl_agent = PolicyAgent()

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "thomasKarloKissing",
        "color": "#499B4A",
        "head": "gamer",
        "tail": "nr-booster",
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")
    rl_agent.load_policy("model/models/latest.pth")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")

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
            for body_part in snake['body'].reverse():
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
                observation[0, 3, snake_length, snake['head']['x'], snake['head']['y']] = 1 if snake_length >= my_length else 0
        
        # Layer 4: food
        for food_item in food:
            observation[0, 4, food_item['x'], food_item['y']] = 1
        
        for x in range(width):
            for y in range(height):
                # Layer 5: gameboard
                observation[0, 5, x, y] = 1

                # Layer 10-16: Alive count
                observation[0, 10 + alive_snakes, x, y] = 1

def move_policy(game_state: typing.Dict) -> typing.Dict:
    """ Use RL agent to get next move """

    # Calculates state observation
    observation = get_observation(game_state)

    # Use RL agent to get action
    action = rl_agent.use_policy(torch.tensor(observation, dtype=torch.float32).to(rl_agent.device))
    
    # TODO: debug action
    print("ACTION: ", action)

    # Map action to direction
    # TODO: check these values
    if action == 0:
        next_move = "up"
    elif action == 1:
        next_move = "down"
    elif action == 2:
        next_move = "left"
    else:
        next_move = "right"

    return {"move": next_move, "shout": GROUP_SHOUT}

# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False

    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False

    # Step 1 - Prevent your Battlesnake from moving out of bounds
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    if my_head["x"] == 0:
        is_move_safe["left"] = False
    elif my_head["x"] == board_width - 1:
        is_move_safe["right"] = False
    if my_head["y"] == 0:
        is_move_safe["down"] = False
    elif my_head["y"] == board_height - 1:
        is_move_safe["up"] = False

    # Step 2 - Prevent your Battlesnake from colliding with itself
    my_body = game_state['you']['body']
    for i in range(1, len(my_body)):
        if my_body[i] == my_head:
            continue
        if my_body[i]["x"] == my_head["x"] - 1 and my_body[i]["y"] == my_head["y"]:
            is_move_safe["left"] = False
        elif my_body[i]["x"] == my_head["x"] + 1 and my_body[i]["y"] == my_head["y"]:
            is_move_safe["right"] = False
        elif my_body[i]["y"] == my_head["y"] - 1 and my_body[i]["x"] == my_head["x"]:
            is_move_safe["down"] = False
        elif my_body[i]["y"] == my_head["y"] + 1 and my_body[i]["x"] == my_head["x"]:
            is_move_safe["up"] = False

    # Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
    opponents = game_state['board']['snakes']
    for opponent in opponents:
        for body_part in opponent['body']:
            if body_part == my_head:
                continue
            if body_part["x"] == my_head["x"] - 1 and body_part["y"] == my_head["y"]:
                is_move_safe["left"] = False
            elif body_part["x"] == my_head["x"] + 1 and body_part["y"] == my_head["y"]:
                is_move_safe["right"] = False
            elif body_part["y"] == my_head["y"] - 1 and body_part["x"] == my_head["x"]:
                is_move_safe["down"] = False
            elif body_part["y"] == my_head["y"] + 1 and body_part["x"] == my_head["x"]:
                is_move_safe["up"] = False

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # Choose a random move from the safe ones
    next_move = random.choice(safe_moves)

    # Step 4 - Move towards food instead of random, to regain health and survive longer
    food = game_state['board']['food']
    closest_food = food[0]
    closest_distance = abs(my_head['x'] - closest_food['x']) + abs(my_head['y'] - closest_food['y'])
    for food_item in food:
        distance = abs(my_head['x'] - food_item['x']) + abs(my_head['y'] - food_item['y'])
        if distance < closest_distance:
            closest_food = food_item
            closest_distance = distance

    if closest_food['x'] < my_head['x'] and "left" in safe_moves:
        next_move = "left"
    elif closest_food['x'] > my_head['x'] and "right" in safe_moves:
        next_move = "right"
    elif closest_food['y'] < my_head['y'] and "down" in safe_moves:
        next_move = "down"
    elif closest_food['y'] > my_head['y'] and "up" in safe_moves:
        next_move = "up"

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move_policy, "end": end})
