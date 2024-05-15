import numpy as np
import json
import model

from sklearn.model_selection import GridSearchCV

HYPERPARAM_FILE = "config/best_hyperparameters.json"

def hyperparam_tuning():
    hyperparameters = {
        'value_loss_coef': [0.25, 0.4, 0.5, 0.6],
        'entropy_coef': [0.001, 0.005, 0.01, 0.025],
        'max_grad_norm': [0.1, 0.25, 0.5, 0.75],
        'clip_param': [0.1, 0.2, 0.3],
        'ppo_epoch': [3, 4, 5],
        'num_mini_batch': [16, 32, 64],
        'eps': [1e-6, 1e-5, 1e-4],
        'lr': [1e-3, 1e-2, 1e-4, 5e-3, 5e-4]
    }

    # hyperparameters = {
    #     'value_loss_coef': [0.5],
    #     'entropy_coef': [0.01],
    #     'max_grad_norm': [0.5],
    #     'clip_param': [0.2],
    #     'ppo_epoch': [4],
    #     'num_mini_batch': [32],
    #     'eps': [1e-5],
    #     'lr': [1e-3]
    # }

    rl_estimator = model.RLEstimator()

    # Create a dummy dataset
    X_dummy = np.zeros((10, 1))
    y_dummy = np.zeros(10)

    grid_search = GridSearchCV(estimator=rl_estimator, param_grid=hyperparameters, cv=3, verbose=2)
    grid_search.fit(X_dummy, y_dummy)

    best_params = grid_search.best_params_

    with open(HYPERPARAM_FILE, "w") as f:
        json.dump(best_params, f)
    print(f"Best hyperparameters saved to {HYPERPARAM_FILE}")

def train_model():
    try:
        with open(HYPERPARAM_FILE, "r") as f:
            best_params = json.load(f)
    except FileNotFoundError:
        print("No hyperparameters file found, selecting default hyperparameters...")
        best_params = {
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'clip_param': 0.2,
            'ppo_epoch': 4,
            'num_mini_batch': 32,
            'eps': 1e-5,
            'lr': 1e-3
        }
    print("Loaded best hyperparameters:", best_params)

    rl_agent = model.RLAgent(50, 400, 50)
    rl_agent.set_agent(best_params.get("value_loss_coef"), best_params.get("entropy_coef"), best_params.get("max_grad_norm"), best_params.get("clip_param"), best_params.get("ppo_epoch"), best_params.get("num_mini_batch"), best_params.get("eps"), best_params.get("lr"))
    
    print("-"*10, "Training model...", "-"*10)
    rl_agent.train()
    print("-"*10, "Training complete!", "-"*10)

    rl_agent.save_results("results")
    rl_agent.save_policy("models")
    rl_agent.save_video("videos")

    print("-"*10, "Run completed!", "-"*10)

if __name__ == "__main__":
    hyperparam_tuning()
    train_model()
