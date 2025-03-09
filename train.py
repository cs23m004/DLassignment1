from neural_network_final import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from parse_args import parse_arguments

def train(xtrain, ytrain, xval, yval, config, verbose=False, seed=None):

  EPOCHS, BATCH_SIZE = config['EPOCHS'], config['BATCH_SIZE']
  lfunc, optim = config['loss_func'], config['optim']
  LR, MOMENTUM, WD = config['LR'], config['MOMENTUM'], config['WD']
  BETA, BETA1, BETA2, EPSILON = config['BETA'], config['BETA1'], config['BETA2'], config['EPSILON']
  W_init, activation, n_hidden = config['W_init'], config['activation'], config['n_hidden']

  
  if seed:
    np.random.seed(seed)

  loss_func, loss_func_der = get_loss_func(lfunc)
  act, act_der = get_activation(activation)
  params = init_params(input_num = xtrain.shape[1], hidden_size = n_hidden, output_num = ytrain.shape[1], init_type= W_init, seed = seed)
  U = {k:0 for k in params.keys()}
  V = {k:0 for k in params.keys()}


  logs = {key: [] for key in ['epochs', 'train_loss', 'train_acc', 'val_loss', 'val_acc']}


  for epoch in range(1, EPOCHS + 1):
    
    ind = np.zeros(ytest.shape[0])
    leng = xtrain.shape[0]
    indices = np.arange(leng)
    np.random.shuffle(indices)

    leng = xtrain.shape[0]
    n_batches = leng // BATCH_SIZE
    ttl_batch = n_batches * BATCH_SIZE
    batches = indices[:ttl_batch].reshape(-1, BATCH_SIZE).tolist()
    
    if xtrain.shape[0] % BATCH_SIZE != 0:
        batches.append(indices[n_batches * BATCH_SIZE:].tolist())

    for batch_idx, batch in enumerate(batches):
        x_batch, y_batch = xtrain[batch, :], ytrain[batch]
        yhat, cache = forward(x_batch, params, act)
        del_params = backward(y_batch, params, yhat, cache, act_der, loss_func_der, WD)

        if optim == 'sgd':
            for k in params:
                params[k] -= LR * del_params[k]

        elif optim == 'momentum':
            for k in params:
                U[k] = MOMENTUM * U[k] + del_params[k]
                params[k] -= LR * U[k]

        elif optim == 'rmsprop':
            for k in params:
                V[k] = BETA * V[k] + (1 - BETA) * (del_params[k] ** 2)
                var = np.sqrt(V[k] + EPSILON)
                params[k] = params[k] - (LR / var) * del_params[k]
        


        elif optim == 'nadam':
            n_updates = (epoch - 1) * len(batches) + (batch_idx + 1)
            
            for k in params:
                U[k] = BETA1 * U[k] + (1 - BETA1) * del_params[k]
                Uk_hat = U[k] / (1 - BETA1 ** n_updates)
                
                V[k] = BETA2 * V[k] + (1 - BETA2) * (del_params[k] ** 2)
                Vk_hat = V[k] / (1 - BETA2 ** n_updates)
                
                params[k] -= (LR / (np.sqrt(Vk_hat) + EPSILON)) * (BETA1 * Uk_hat + (1 - BETA1) * del_params[k] / (1 - BETA1 ** n_updates))
        
        elif optim == 'nag':
            params_LA = {k: params[k] - MOMENTUM * U[k] for k in params}
            yhat, cache = forward(x_batch, params_LA, act)
            del_params = backward(y_batch, params_LA, yhat, cache, act_der, loss_func_der, WD)
            
            for k in params:
                U[k] = MOMENTUM * U[k] + del_params[k]
                params[k] -= LR * U[k]

        elif optim == 'adam':
            n_updates = (epoch - 1) * len(batches) + (batch_idx + 1)
            
            for k in params:
                U[k] = BETA1 * U[k] + (1 - BETA1) * del_params[k]
                Uk_hat = U[k] / (1 - BETA1 ** n_updates)
                
                V[k] = BETA2 * V[k] + (1 - BETA2) * (del_params[k] ** 2)
                Vk_hat = V[k] / (1 - BETA2 ** n_updates)
                
                params[k] -= (LR / (np.sqrt(Vk_hat) + EPSILON)) * Uk_hat


        else:
            raise Exception('Error : Wrong Optimizer value')

    train_loss, train_acc = eval_params(xtrain, ytrain, params, config)
    val_loss, val_acc = eval_params(xval, yval, params, config)

    logs.update({
        'epochs': logs['epochs'] + [epoch],
        'train_loss': logs['train_loss'] + [train_loss],
        'train_acc': logs['train_acc'] + [train_acc],
        'val_loss': logs['val_loss'] + [val_loss],
        'val_acc': logs['val_acc'] + [val_acc]
    })

    if verbose:
        print(f"Epoch {epoch}:: Training: Loss = {train_loss:.4f} Accuracy = {train_acc:.4f}  "
              f"Validation: Loss = {val_loss:.4f} Accuracy = {val_acc:.4f}")

  return params, logs

if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_arguments()

    # Configure model hyperparameters
    config = {
        'EPOCHS': args.epochs,
        'loss_func': args.loss,
        'optim': args.optimizer,
        'BATCH_SIZE': args.batch_size,
        'n_hidden': [args.hidden_size] * args.num_layers,
        'WD': args.weight_decay,
        'activation': args.activation,
        'LR': args.learning_rate,
        'MOMENTUM': args.momentum,
        'BETA': args.beta,
        'BETA1': args.beta1,
        'BETA2': args.beta2,
        'EPSILON': args.epsilon,
        'W_init': args.weight_init
    }

    # Load and preprocess dataset
    dataset = get_dataset(args.dataset)
    (xtrain, ytrain), (xval, yval), (xtest, ytest), class_labels = dataset

    scaled_dataset = scale_dataset(xtrain, ytrain, xval, yval, xtest, ytest, args.dataset_scaling)
    (xtrain_inp, ytrain_inp), (xval_inp, yval_inp), (xtest_inp, ytest_inp) = scaled_dataset

    # Train the model
    params, logs = train(
        xtrain=xtrain_inp, ytrain=ytrain_inp,
        xval=xval_inp, yval=yval_inp,
        config=config, verbose=True, seed=0
    )


    # Generate predictions
    predictions = {split: predict(data, params, config) for split, data in 
                   zip(['train', 'val', 'test'], [xtrain_inp, xval_inp, xtest_inp])}

    ytrain_hat, yval_hat, ytest_hat = predictions.values()

    # Evaluate model performance
    train_loss, train_acc = eval_params(xtrain_inp, ytrain_inp, params, config)
    val_loss, val_acc = eval_params(xval_inp, yval_inp, params, config)
    test_loss, test_acc = eval_params(xtest_inp, ytest_inp, params, config)

    # Compute confusion matrices
    confusion_matrices = {split: confusion_matrix(true, pred) for split, true, pred in 
                          zip(['train', 'val', 'test'], [ytrain, yval, ytest], [ytrain_hat, yval_hat, ytest_hat])}

    CM_train, CM_val, CM_test = confusion_matrices.values()


    # Define class labels for visualization
    _class_labels = [class_labels[k] for k in range(10)]

    # Function to plot confusion matrices
    def plot_confusion_matrix(cm, title):
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, xticklabels=_class_labels, yticklabels=_class_labels,
                    fmt='g', annot_kws={"fontsize": 6}, cmap='Greens')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.title(title)

    # Plot and display confusion matrices
    plot_confusion_matrix(CM_train, 'Train Data Confusion Matrix')
    plot_confusion_matrix(CM_val, 'Validation Data Confusion Matrix')
    plot_confusion_matrix(CM_test, 'Test Data Confusion Matrix')

    # Log results to Weights & Biases if enabled
    if args.wandb_log == 'True':
        import wandb
        wandb.login()
        
        run = wandb.init(
            entity=args.wandb_entity, 
            project=args.wandb_project, 
            name=args.wandb_name
        )

        # Log training metrics
        for epoch, train_loss, train_acc, val_loss, val_acc in zip(
            logs['epochs'], logs['train_loss'], logs['train_acc'], logs['val_loss'], logs['val_acc']
        ):
            wandb.log({
                'epochs': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

        # Log confusion matrices
        wandb.log({'CM_train': wandb.Image(plot_confusion_matrix(CM_train, 'Train Data Confusion Matrix'))})
        wandb.log({'CM_val': wandb.Image(plot_confusion_matrix(CM_val, 'Validation Data Confusion Matrix'))})
        wandb.log({'CM_test': wandb.Image(plot_confusion_matrix(CM_test, 'Test Data Confusion Matrix'))})

        # Log final evaluation metrics
        wandb.log({'Train Accuracy': train_acc})
        wandb.log({'Validation Accuracy': val_acc})
        wandb.log({'Test Accuracy': test_acc})

        wandb.finish()

    # Display evaluation results
    print("\nModel Evaluation:")
    print(f"Training: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
    print(f"Validation: Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}")
    print(f"Testing: Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}\n")

    # Print confusion matrices
    print("Confusion Matrix of Training Data:\n", CM_train, "\n")
    print("Confusion Matrix of Validation Data:\n", CM_val, "\n")
    print("Confusion Matrix of Test Data:\n", CM_test, "\n")

    plt.show()