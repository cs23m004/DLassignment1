import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Neural Network Training Configuration")

    # WandB Logging Arguments
    parser.add_argument('-wp', '--wandb_project', type=str, default='Assignment1', 
                        help='Project name on WandB')
    parser.add_argument('-we', '--wandb_entity', type=str, 
                        default='cs23m004-indian-institute-of-technology-madras', 
                        help='Username on WandB')
    parser.add_argument('-wn', '--wandb_name', type=str, default='cs23m004', 
                        help='Display name of run on WandB')
    parser.add_argument('-wl', '--wandb_log', type=lambda x: x.lower() == 'true', 
                        default=False, help='Enable WandB logging (True/False)')

    # Dataset and Training Parameters
    parser.add_argument('-d', '--dataset', type=str, choices=['fashion_mnist', 'mnist'], 
                        default='fashion_mnist', help='Dataset to use')
    parser.add_argument('-e', '--epochs', type=int, default=30, 
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=128, 
                        help='Batch size for training')

    # Loss and Optimizer Configuration
    parser.add_argument('-l', '--loss', type=str, choices=['cross_entropy', 'mean_squared_error'], 
                        default='cross_entropy', help='Loss function to use')
    parser.add_argument('-o', '--optimizer', type=str, 
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], 
                        default='adam', help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, 
                        help='Learning rate')

    # Optimizer Hyperparameters
    parser.add_argument('-m', '--momentum', type=float, default=0.9, 
                        help='Momentum for "momentum" and "nag" optimizers')
    parser.add_argument('-beta', '--beta', type=float, default=0.9, 
                        help='Beta for RMSprop optimizer')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.8, 
                        help='Beta1 for Adam and Nadam optimizers')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999, 
                        help='Beta2 for Adam and Nadam optimizers')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6, 
                        help='Epsilon for numerical stability in optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.3, 
                        help='Weight decay (L2 regularization)')

    # Model Architecture
    parser.add_argument('-w_i', '--weight_init', type=str, choices=['random', 'Xavier'], 
                        default='Xavier', help='Weight initialization strategy')
    parser.add_argument('-nhl', '--num_layers', type=int, default=5, 
                        help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=128, 
                        help='Number of neurons per hidden layer')
    parser.add_argument('-a', '--activation', type=str, 
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'], 
                        default='ReLU', help='Activation function')

    # Data Scaling
    parser.add_argument('-ds', '--dataset_scaling', type=str, 
                        choices=['min_max', 'standard'], default='standard', 
                        help='Scaling method for the dataset')

    return parser.parse_args()
