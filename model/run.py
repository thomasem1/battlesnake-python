import numpy as np
import json
import model

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

SAVE_FILE = "best_hyperparameters.json"

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
        self.rl_agent = model.RLAgent(4, 4, 4)

    def fit(self, X, y=None):
        # Set the agent's hyperparameters
        self.rl_agent.set_agent(self.value_loss_coef, self.entropy_coef, self.max_grad_norm,
                                self.clip_param, self.ppo_epoch, self.num_mini_batch,
                                self.eps, self.lr)
        # Train the agent
        print(f"value loss coef {self.value_loss_coef}, entropy coef {self.entropy_coef}, max grad norm {self.max_grad_norm}, clip param {self.clip_param}, ppo epoch {self.ppo_epoch}, num mini batch {self.num_mini_batch}, eps {self.eps}, lr {self.lr}")
        self.rl_agent.train()
        return self

    def score(self, X, y=None):
        # Evaluate the agent and return the negative of the episode length
        # (since GridSearchCV tries to maximize the score)
        return self.rl_agent.get_episode_length()

rl_estimator = RLEstimator()

hyperparameters = {
    'value_loss_coef': [0.1, 0.25, 0.5, 1.0],
    'entropy_coef': [0.001, 0.005, 0.01, 0.1],
    'max_grad_norm': [0.1, 0.25, 0.5, 1.0],
    'clip_param': [0.1, 0.2, 0.3],
    'ppo_epoch': [3, 4, 5],
    'num_mini_batch': [32, 64],
    'eps': [1e-6, 1e-5, 1e-4],
    'lr': [0.0001, 0.0005, 0.001, 0.005, 0.01]
}

# Create a dummy dataset
X_dummy = np.zeros((10, 1))
y_dummy = np.zeros(10)

grid_search = GridSearchCV(estimator=rl_estimator, param_grid=hyperparameters, cv=3, verbose=2)
grid_search.fit(X_dummy, y_dummy)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)


with open(SAVE_FILE, "w") as f:
    json.dump(best_params, f)
print(f"Best hyperparameters saved to {SAVE_FILE}")
