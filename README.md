## DLassignment1 -
# About Content :
neural_network_final.py : This file contains implementation of neural network using numpy along with several helper functions.Functions such as forward pass, back propagation, helping functions such as activation funcitons, weight initialisation etc are implemented in this python file.

train.py : This function is responsible for training the neural network with given parameters. Use semantics of the file are provided below.

pars_args.py : Python script to handle command-line arguments for the train.py file.

# Train.py - Command Line Arguments

This script allows users to train a model using various hyperparameters and configurations.

## Execution

Run the following command in the terminal:

```bash

python3 train.py [-h --help] 
                        [-wp --wandb_project] <string>
                        [-we --wandb_entity] <string>
                        [-wn --wandb_name] <string>
                        [-wl --wandb_log] <"True", "False">
                        [-d --dataset] <"fashion_mnist", "mnist">
                        [-e --epochs] <int>
                        [-b --batch_size] <int>
                        [-l --loss] <"cross_entropy", "mean_squared_error">
                        [-o --optimizer] <"sgd", "momentum", "nag", "rmsprop", "adam", "nadam">
                        [-lr --learning_rate] <float>
                        [-m --momentum] <float>
                        [-beta --beta] <float>
                        [-beta1 --beta1] <float>
                        [-beta2 --beta2] <float>
                        [-eps --epsilon] <float>
                        [-w_d --weight_decay] <float>
                        [-w_i --weight_init] <"random", "Xavier">
                        [-nhl --num_layers] <int>
                        [-sz --hidden_size] <int>
                        [-a --activation] <"identity", "sigmoid", "tanh", "ReLU">
                        [-ds --data_scaling] <"min_max", "standard">       	
