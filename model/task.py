import argparse
import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with PPO algorithm')

    parser.add_argument('--n_envs', type=int, default=50, help='Number of environments')
    parser.add_argument('--n_steps', type=int, default=400, help='Number of steps per environment')
    parser.add_argument('--num_updates', type=int, default=50, help='Number of updates')
    parser.add_argument('--value_loss_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--clip_param', type=float, default=0.2, help='Clip parameter')
    parser.add_argument('--ppo_epoch', type=int, default=4, help='PPO epoch')
    parser.add_argument('--num_mini_batch', type=int, default=32, help='Number of mini-batches')
    parser.add_argument('--eps', type=float, default=1e-5, help='Epsilon')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    args = parser.parse_args()

    model.train_model(args.n_envs, args.n_steps, args.num_updates, args.value_loss_coef, args.entropy_coef, args.max_grad_norm,
                args.clip_param, args.ppo_epoch, args.num_mini_batch, args.eps, args.lr)

