"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import pandas as pd
import numpy as np
import random

DEVICE = torch.device("cuda:0")
BATCH_SIZE = 128
CLASSES = 3
DIR = os.getcwd()
EPOCHS = 1000
# N_TRAIN_EXAMPLES = BATCH_SIZE * 30
# N_VALID_EXAMPLES = BATCH_SIZE * 10


#     return total_loss
def custom_loss_function(outputs, labels, inputs, criterion, alpha, tolerance=1e-5):
    # Calculate the original MSE loss
    loss = criterion(outputs, labels)

    # Extract inputs and outputs
    pA, pB, dA, dB = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    IB, IA = outputs[:, 0], outputs[:, 1]

    # Identify constraint indices with a soft tolerance
    constraint_indices = torch.logical_and(
        torch.abs(pA - pB) < tolerance,
        torch.abs(dA - dB) < tolerance
    )

    # Calculate penalty for IB != IA
    if constraint_indices.sum() > 0:
        penalty = torch.mean((IB[constraint_indices] - IA[constraint_indices]) ** 2)
    else:
        penalty = torch.tensor(0.0, device=outputs.device)
    num_constraints = constraint_indices.sum().item()
    penalty_weight = num_constraints / len(inputs)  # Proportional weight

    # Dynamically scale penalty relative to MSE
    # penalty_weight = penalty.item()/(loss.item() + 1e-8)
    total_loss = loss + alpha * penalty_weight * penalty


    return total_loss
    

def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    in_features = 5
    for i in range(n_layers):
        # out_features = trial.suggest_int("n_units_l{}".format(i), 1, 256)
        out_features = trial.suggest_categorical(f"n_units_l{i}", [32, 64, 128, 256])
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        # p = trial.suggest_float("dropout_l{}".format(i), 0, 0)
        # layers.append(nn.Dropout(p))

        in_features = out_features
        
    layers.append(nn.Linear(in_features, CLASSES))
    # layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

def get_dataset():
    # Load .json Files
    DATA = pd.read_csv('normalized_large.csv')        
    
    pA= DATA.loc[:,'pA'].values.reshape((-1,1))
    pB = DATA.loc[:,'pB'].values.reshape((-1,1))
    dA = DATA.loc[:,'dA'].values.reshape((-1,1))
    dB = DATA.loc[:,'dB'].values.reshape((-1,1))
    dP = DATA.loc[:,'dP'].values.reshape((-1,1))
    IB = DATA.loc[:,'IB'].values.reshape((-1,1))
    IA = DATA.loc[:,'IA'].values.reshape((-1,1))
    IP = DATA.loc[:,'IP'].values.reshape((-1,1))

    # Compute labels
    # f = f.reshape((-1,1))
    # Lr = Lr.reshape((-1,1))
    # Cr = Cr.reshape((-1,1))
    # Lm = Lm.reshape((-1,1))
    # # fr = fr.reshape((-1,1))
    # R = R.reshape((-1,1))
    # Vo = Vo.reshape((-1,1))


    temp_input = np.concatenate((pA,pB,dA,dB,dP),axis=1)
    temp_output = np.concatenate((IB,IA,IP),axis=1)
    
    in_tensors = torch.from_numpy(temp_input).view(-1, 5)
    out_tensors = torch.from_numpy(temp_output).view(-1, 3)

    # # Save dataset for future use
    np.save("dataset.fc.in.npy", in_tensors.numpy())
    np.save("dataset.fc.out.npy", out_tensors.numpy())

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)



def get_dataloader():
    # Load dataset
    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    dataset = get_dataset()

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset)-train_size

    # train_size = N_TRAIN_EXAMPLES
    # valid_size = N_VALID_EXAMPLES


    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda:0"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    return train_loader, valid_loader, train_size, valid_size



def objective(trial):
    # Suggest hyperparameters for the model, optimizer, alpha, and batch size
    alpha = trial.suggest_float("alpha", 0.0001, 1, log=True)  # Range for alpha
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])  # Limited batch sizes

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    optimizer_name = "Adam"
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)

    # Get the dataset and update the DataLoader with dynamic batch size
    dataset = get_dataset()
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    DECAY_RATIO = 0.5
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr*(DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH)))

    # # Get the FashionMNIST dataset.
    # train_loader, valid_loader, train_size, valid_size = get_dataloader()
    # Training of the model.
    for epoch in range(EPOCHS):
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr * (0.5 ** (epoch // 100)))
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data.float())
            criterion = nn.MSELoss()
            # Use the suggested alpha in the custom loss function
            loss = custom_loss_function(output, target.float(), data.float(), criterion, alpha=alpha)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        epoch_valid_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data.float())
                criterion = nn.MSELoss()
                loss = custom_loss_function(output, target.float(), data.float(), criterion, alpha=alpha)
                epoch_valid_loss += loss.item()

        # Compute the validation loss
        accuracy = epoch_valid_loss / len(valid_loader)

        # Report to Optuna
        trial.report(accuracy, epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1000, timeout=60000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
