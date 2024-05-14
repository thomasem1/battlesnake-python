import numpy as np
import json
import model

from sklearn.model_selection import GridSearchCV

def run():
    rl_estimator = model.RLEstimator()

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

    hyperparameters = {
        'value_loss_coef': [0.1, 0.25, 0.5, 1.0],
        'entropy_coef': [0.1],
        'max_grad_norm': [1.0],
        'clip_param': [0.3],
        'ppo_epoch': [5],
        'num_mini_batch': [32],
        'eps': [1e-4],
        'lr': [0.01]
    }

    # Create a dummy dataset
    X_dummy = np.zeros((10, 1))
    y_dummy = np.zeros(10)

    grid_search = GridSearchCV(estimator=rl_estimator, param_grid=hyperparameters, cv=3, verbose=2)
    grid_search.fit(X_dummy, y_dummy)

    best_params = grid_search.best_params_

    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_params, f)
    print("Best hyperparameters saved to best_hyperparameters.json")

if __name__ == "__main__":
    run()