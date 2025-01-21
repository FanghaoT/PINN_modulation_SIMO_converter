# Import necessary packages
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import json
import math
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 143),
            nn.ReLU(),
            nn.Linear(143, 169),
            nn.ReLU(),
            nn.Linear(169, 89),
            nn.ReLU(),
            nn.Linear(89, 3),
        )

    def forward(self, x):
        return self.layers(x)
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataset(data_path):
    # Load .json Files
    DATA = pd.read_csv(data_path)        
    
    pA= DATA.loc[:,'pA'].values.reshape((-1,1))
    pB = DATA.loc[:,'pB'].values.reshape((-1,1))
    dA = DATA.loc[:,'dA'].values.reshape((-1,1))
    dB = DATA.loc[:,'dB'].values.reshape((-1,1))
    dP = DATA.loc[:,'dP'].values.reshape((-1,1))
    IB = DATA.loc[:,'IB'].values.reshape((-1,1))
    IA = DATA.loc[:,'IA'].values.reshape((-1,1))
    IP = DATA.loc[:,'IP'].values.reshape((-1,1))


    temp_input = np.concatenate((pA,pB,dA,dB,dP),axis=1)
    temp_output = np.concatenate((IB,IA,IP),axis=1)
    
    in_tensors = torch.from_numpy(temp_input).view(-1, 5)
    out_tensors = torch.from_numpy(temp_output).view(-1, 3)

    # # Save dataset for future use
    np.save("dataset.fc.in.npy", in_tensors.numpy())
    np.save("dataset.fc.out.npy", out_tensors.numpy())

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)


#     return total_loss
def custom_loss_function(outputs, labels, inputs, criterion, alpha=0.0008376104970584724, tolerance=1e-5):
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
    


# Config the model training

def main():

    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)
    # Hyperparameters
    NUM_EPOCH = 1000
    # BATCH_SIZE = 128
    BATCH_SIZE = 64
    DECAY_EPOCH = 100
    DECAY_RATIO = 0.5
    LR_INI = 0.0009493094423558486
    # Select GPU as default device
    device = torch.device("cuda:0")

    train_data_path='normalized_large.csv'
    test_data_path='normalized_test.csv'
    test_same_data_path='normalized_test_same.csv'
    test_other_data_path='normalized_test_other.csv'
    # Load dataset
    dataset = get_dataset(train_data_path)
    test_dataset = get_dataset(test_data_path)
    test_same_dataset = get_dataset(test_same_data_path)
    test_other_dataset = get_dataset(test_other_data_path)

    # Split the dataset
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset)-train_size
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda:0"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_same_loader = torch.utils.data.DataLoader(test_same_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_other_loader = torch.utils.data.DataLoader(test_other_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    # Setup network
    net = Net().double().to(device)

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 

    # train_loss_list = np.zeros(NUM_EPOCH)
    # valid_loss_list = np.zeros(NUM_EPOCH)
        #define a list to store the validating loss in dimension of epoch*5
    valid_loss_list = np.zeros([NUM_EPOCH,4])

    #define a list to store the training loss in dimension of epoch*5
    train_loss_list = np.zeros([NUM_EPOCH,4])

    count_loss=0
    # Train the network
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        epoch_train_loss1 = 0
        epoch_train_loss2 = 0
        epoch_train_loss3 = 0
        # epoch_train_loss4 = 0


        net.train()
        optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))
        # optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i))      
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            # loss = criterion(outputs, labels.to(device))
            loss = custom_loss_function(outputs, labels.to(device), inputs.to(device), criterion)
            loss1 = criterion(outputs[:,0], labels[:,0].to(device))
            loss2 = criterion(outputs[:,1], labels[:,1].to(device))
            loss3 = criterion(outputs[:,2], labels[:,2].to(device))
            # loss4 = criterion(outputs[:,3], labels[:,3].to(device))
            loss.backward()
            optimizer.step()


            # print(loss.item())

            epoch_train_loss += loss.item()
            epoch_train_loss1 += loss1.item()
            epoch_train_loss2 += loss2.item()
            epoch_train_loss3 += loss3.item()
            # epoch_train_loss4 += loss4.item()


            count_loss+=1
            
            
        # Compute Validation Loss
        with torch.no_grad():
            epoch_valid_loss = 0
            epoch_valid_loss1 = 0
            epoch_valid_loss2 = 0
            epoch_valid_loss3 = 0
            # epoch_valid_loss4 = 0

            for inputs, labels in valid_loader:
                outputs = net(inputs.to(device))
                # loss = criterion(outputs, labels.to(device))
                loss = custom_loss_function(outputs, labels.to(device), inputs.to(device), criterion)
                loss1 = criterion(outputs[:,0], labels[:,0].to(device))
                loss2 = criterion(outputs[:,1], labels[:,1].to(device))
                loss3 = criterion(outputs[:,2], labels[:,2].to(device))
                # loss4 = criterion(outputs[:,3], labels[:,3].to(device))

                epoch_valid_loss += loss.item()
                epoch_valid_loss1 += loss1.item()
                epoch_valid_loss2 += loss2.item()
                epoch_valid_loss3 += loss3.item()
                # epoch_valid_loss4 += loss4.item()

        #Save the training and validation loss into a list.
        train_loss_list[epoch_i] = [epoch_train_loss / len(train_loader), epoch_train_loss1 / len(train_loader), epoch_train_loss2 / len(train_loader), epoch_train_loss3 / len(train_loader)]
        valid_loss_list[epoch_i] = [epoch_valid_loss / len(valid_loader), epoch_valid_loss1 / len(valid_loader), epoch_valid_loss2 / len(valid_loader), epoch_valid_loss3 / len(valid_loader)]
        # train_loss_list[epoch_i] = [epoch_train_loss / len(train_loader)]
        # valid_loss_list[epoch_i] = [epoch_valid_loss / len(valid_loader)]
        if (epoch_i+1)%200 == 0:
          print(f"Epoch {epoch_i+1:2d} "
              f"Train {epoch_train_loss / len(train_loader):.10f} "
              f"Valid {epoch_valid_loss / len(valid_loader):.10f}")

    print(count_loss)    
    # Save the model parameters
    torch.save(net.state_dict(), "Model_FNN.sd")
    print("Training finished! Model is saved!")
    
    np.savetxt('train_loss_overall.csv',train_loss_list,delimiter=',')
    np.savetxt('valid_loss_overall.csv',valid_loss_list,delimiter=',')
    # Evaluation


    #
    print('This is the test results for all the data')
    net.eval()
    inputs_list = []
    y_meas = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = net(inputs.to(device))
            y_pred.append(outputs)
            y_meas.append(labels.to(device))
            inputs_list.append(inputs.to(device))

    # Concatenate all batches
    inputs_array = torch.cat(inputs_list, dim=0)
    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    # Calculate overall metrics
    mse_value = F.mse_loss(y_meas, y_pred).item()
    rmse_value = np.sqrt(mse_value)
    mae_value = F.l1_loss(y_meas, y_pred).item()
    print(f"Test MSE: {mse_value:.10f}")
    print(f"Test RMSE: {rmse_value:.10f}")
    print(f"Test MAE: {mae_value:.10f}")

    # Convert tensors to numpy arrays for further processing
    inputs_array = inputs_array.cpu().numpy()
    y_meas = y_meas.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    # Load normalization boundaries
    boundaries = pd.read_csv("boundary_large.csv")
    min_vals = boundaries['min'].values
    max_vals = boundaries['max'].values

    # Function to reverse normalize data
    def reverse_normalize(data, min_val, max_val):
        return data * (max_val - min_val) + min_val

    # Reverse normalization for inputs and outputs
    for i in range(inputs_array.shape[1]):
        inputs_array[:, i] = reverse_normalize(inputs_array[:, i], min_vals[i], max_vals[i])

    for i in range(y_meas.shape[1]):
        y_meas[:, i] = reverse_normalize(y_meas[:, i], min_vals[i + 5], max_vals[i + 5])
        y_pred[:, i] = reverse_normalize(y_pred[:, i], min_vals[i + 5], max_vals[i + 5])

    # DataFrame to save
    df = pd.DataFrame(inputs_array, columns=[f'input_{i}' for i in range(inputs_array.shape[1])])
    output_data = {'y_meas_0': y_meas[:, 0], 'y_pred_0': y_pred[:, 0], 
                'y_meas_1': y_meas[:, 1], 'y_pred_1': y_pred[:, 1],
                'y_meas_2': y_meas[:, 2], 'y_pred_2': y_pred[:, 2]}
    df = pd.concat([df, pd.DataFrame(output_data)], axis=1)


    epsilon = 1e-8
    metrics = {'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'R2': []}

    # Calculate errors for each sample
    for i in range(3):
        mae = np.abs(y_meas[:, i] - y_pred[:, i])
        mse = (y_meas[:, i] - y_pred[:, i]) ** 2
        rmse = np.sqrt(mse)
        mape = np.abs((y_meas[:, i] - y_pred[:, i]) / (y_meas[:, i] + epsilon)) * 100
        r2 = r2_score(y_meas[:, i], y_pred[:, i])

        # Store each metric in the DataFrame
        df[f'MAE_{i}'] = mae
        df[f'MSE_{i}'] = mse
        df[f'RMSE_{i}'] = rmse
        df[f'MAPE_{i}'] = mape
        df[f'R2_{i}'] = r2
        # Collect mean values in a dictionary to print later
        metrics['MAE'].append(np.mean(mae))
        metrics['MSE'].append(np.mean(mse))
        metrics['RMSE'].append(np.mean(rmse))
        metrics['MAPE'].append(np.mean(mape))
        metrics['R2'].append(r2)  # R2 is already a summary metric

    # Print mean of each metric for all outputs
    for metric, values in metrics.items():
        print(f"Mean {metric}: {np.mean(values):.10f}")

    # Save the DataFrame to a CSV file
    csv_file_path = 'full_data_y_meas_y_pred.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"Saved full data with measurements and predictions to '{csv_file_path}'")


    # Assuming other parts of the script remain unchanged

    # Your DataFrame 'df' has been already created and populated with values
    # Let's generate scatter plots for each output variable

    output_columns = ['y_meas_0', 'y_pred_0', 'y_meas_1', 'y_pred_1', 'y_meas_2', 'y_pred_2']
    colors = ['red', 'blue', 'green']

    for i in range(3):
        plt.figure(figsize=(10, 6))
        plt.scatter(df[f'y_meas_{i}'], df[f'y_pred_{i}'], color=colors[i], alpha=0.5)
        plt.title(f'Actual vs. Predicted Values for Output {i}\nR^2 Score: {df[f"R2_{i}"].iloc[0]:.4f}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.axis('equal')
        plt.plot([df[f'y_meas_{i}'].min(), df[f'y_meas_{i}'].max()], [df[f'y_meas_{i}'].min(), df[f'y_meas_{i}'].max()], 'k--')  # Diagonal line
        plt.savefig(f'output_{i}_scatter_plot.png')  # Save plot to file
        # plt.show()

#
    print('This is the test results for all the same AB data')
    net.eval()
    inputs_list = []
    y_meas = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_same_loader:
            outputs = net(inputs.to(device))
            y_pred.append(outputs)
            y_meas.append(labels.to(device))
            inputs_list.append(inputs.to(device))

    # Concatenate all batches
    inputs_array = torch.cat(inputs_list, dim=0)
    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    # Calculate overall metrics
    mse_value = F.mse_loss(y_meas, y_pred).item()
    rmse_value = np.sqrt(mse_value)
    mae_value = F.l1_loss(y_meas, y_pred).item()
    print(f"Test MSE: {mse_value:.10f}")
    print(f"Test RMSE: {rmse_value:.10f}")
    print(f"Test MAE: {mae_value:.10f}")

    # Convert tensors to numpy arrays for further processing
    inputs_array = inputs_array.cpu().numpy()
    y_meas = y_meas.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    # Load normalization boundaries
    boundaries = pd.read_csv("boundary_large.csv")
    min_vals = boundaries['min'].values
    max_vals = boundaries['max'].values

    # Function to reverse normalize data
    def reverse_normalize(data, min_val, max_val):
        return data * (max_val - min_val) + min_val

    # Reverse normalization for inputs and outputs
    for i in range(inputs_array.shape[1]):
        inputs_array[:, i] = reverse_normalize(inputs_array[:, i], min_vals[i], max_vals[i])

    for i in range(y_meas.shape[1]):
        y_meas[:, i] = reverse_normalize(y_meas[:, i], min_vals[i + 5], max_vals[i + 5])
        y_pred[:, i] = reverse_normalize(y_pred[:, i], min_vals[i + 5], max_vals[i + 5])

    # DataFrame to save
    df = pd.DataFrame(inputs_array, columns=[f'input_{i}' for i in range(inputs_array.shape[1])])
    output_data = {'y_meas_0': y_meas[:, 0], 'y_pred_0': y_pred[:, 0], 
                'y_meas_1': y_meas[:, 1], 'y_pred_1': y_pred[:, 1],
                'y_meas_2': y_meas[:, 2], 'y_pred_2': y_pred[:, 2]}
    df = pd.concat([df, pd.DataFrame(output_data)], axis=1)


    epsilon = 1e-8
    metrics = {'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'R2': []}

    # Calculate errors for each sample
    for i in range(3):
        mae = np.abs(y_meas[:, i] - y_pred[:, i])
        mse = (y_meas[:, i] - y_pred[:, i]) ** 2
        rmse = np.sqrt(mse)
        mape = np.abs((y_meas[:, i] - y_pred[:, i]) / (y_meas[:, i] + epsilon)) * 100
        r2 = r2_score(y_meas[:, i], y_pred[:, i])

        # Store each metric in the DataFrame
        df[f'MAE_{i}'] = mae
        df[f'MSE_{i}'] = mse
        df[f'RMSE_{i}'] = rmse
        df[f'MAPE_{i}'] = mape
        df[f'R2_{i}'] = r2
        # Collect mean values in a dictionary to print later
        metrics['MAE'].append(np.mean(mae))
        metrics['MSE'].append(np.mean(mse))
        metrics['RMSE'].append(np.mean(rmse))
        metrics['MAPE'].append(np.mean(mape))
        metrics['R2'].append(r2)  # R2 is already a summary metric

    # Print mean of each metric for all outputs
    for metric, values in metrics.items():
        print(f"Mean {metric}: {np.mean(values):.10f}")

    # Save the DataFrame to a CSV file
    csv_file_path = 'full_data_y_meas_y_pred_same.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"Saved full data with measurements and predictions to '{csv_file_path}'")


    # Assuming other parts of the script remain unchanged

    # Your DataFrame 'df' has been already created and populated with values
    # Let's generate scatter plots for each output variable

    output_columns = ['y_meas_0', 'y_pred_0', 'y_meas_1', 'y_pred_1', 'y_meas_2', 'y_pred_2']
    colors = ['red', 'blue', 'green']

    for i in range(3):
        plt.figure(figsize=(10, 6))
        plt.scatter(df[f'y_meas_{i}'], df[f'y_pred_{i}'], color=colors[i], alpha=0.5)
        plt.title(f'Actual vs. Predicted Values for Output {i}\nR^2 Score: {df[f"R2_{i}"].iloc[0]:.4f}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.axis('equal')
        plt.plot([df[f'y_meas_{i}'].min(), df[f'y_meas_{i}'].max()], [df[f'y_meas_{i}'].min(), df[f'y_meas_{i}'].max()], 'k--')  # Diagonal line
        plt.savefig(f'output_{i}_scatter_plot_same.png')  # Save plot to file


    print('This is the test results for all the same AB data')
    net.eval()
    inputs_list = []
    y_meas = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_other_loader:
            outputs = net(inputs.to(device))
            y_pred.append(outputs)
            y_meas.append(labels.to(device))
            inputs_list.append(inputs.to(device))

    # Concatenate all batches
    inputs_array = torch.cat(inputs_list, dim=0)
    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    # Calculate overall metrics
    mse_value = F.mse_loss(y_meas, y_pred).item()
    rmse_value = np.sqrt(mse_value)
    mae_value = F.l1_loss(y_meas, y_pred).item()
    print(f"Test MSE: {mse_value:.10f}")
    print(f"Test RMSE: {rmse_value:.10f}")
    print(f"Test MAE: {mae_value:.10f}")

    # Convert tensors to numpy arrays for further processing
    inputs_array = inputs_array.cpu().numpy()
    y_meas = y_meas.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    # Load normalization boundaries
    boundaries = pd.read_csv("boundary_large.csv")
    min_vals = boundaries['min'].values
    max_vals = boundaries['max'].values

    # Function to reverse normalize data
    def reverse_normalize(data, min_val, max_val):
        return data * (max_val - min_val) + min_val

    # Reverse normalization for inputs and outputs
    for i in range(inputs_array.shape[1]):
        inputs_array[:, i] = reverse_normalize(inputs_array[:, i], min_vals[i], max_vals[i])

    for i in range(y_meas.shape[1]):
        y_meas[:, i] = reverse_normalize(y_meas[:, i], min_vals[i + 5], max_vals[i + 5])
        y_pred[:, i] = reverse_normalize(y_pred[:, i], min_vals[i + 5], max_vals[i + 5])

    # DataFrame to save
    df = pd.DataFrame(inputs_array, columns=[f'input_{i}' for i in range(inputs_array.shape[1])])
    output_data = {'y_meas_0': y_meas[:, 0], 'y_pred_0': y_pred[:, 0], 
                'y_meas_1': y_meas[:, 1], 'y_pred_1': y_pred[:, 1],
                'y_meas_2': y_meas[:, 2], 'y_pred_2': y_pred[:, 2]}
    df = pd.concat([df, pd.DataFrame(output_data)], axis=1)


    epsilon = 1e-8
    metrics = {'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'R2': []}

    # Calculate errors for each sample
    for i in range(3):
        mae = np.abs(y_meas[:, i] - y_pred[:, i])
        mse = (y_meas[:, i] - y_pred[:, i]) ** 2
        rmse = np.sqrt(mse)
        mape = np.abs((y_meas[:, i] - y_pred[:, i]) / (y_meas[:, i] + epsilon)) * 100
        r2 = r2_score(y_meas[:, i], y_pred[:, i])

        # Store each metric in the DataFrame
        df[f'MAE_{i}'] = mae
        df[f'MSE_{i}'] = mse
        df[f'RMSE_{i}'] = rmse
        df[f'MAPE_{i}'] = mape
        df[f'R2_{i}'] = r2
        # Collect mean values in a dictionary to print later
        metrics['MAE'].append(np.mean(mae))
        metrics['MSE'].append(np.mean(mse))
        metrics['RMSE'].append(np.mean(rmse))
        metrics['MAPE'].append(np.mean(mape))
        metrics['R2'].append(r2)  # R2 is already a summary metric

    # Print mean of each metric for all outputs
    for metric, values in metrics.items():
        print(f"Mean {metric}: {np.mean(values):.10f}")

    # Save the DataFrame to a CSV file
    csv_file_path = 'full_data_y_meas_y_pred_other.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"Saved full data with measurements and predictions to '{csv_file_path}'")


    # Assuming other parts of the script remain unchanged

    # Your DataFrame 'df' has been already created and populated with values
    # Let's generate scatter plots for each output variable

    output_columns = ['y_meas_0', 'y_pred_0', 'y_meas_1', 'y_pred_1', 'y_meas_2', 'y_pred_2']
    colors = ['red', 'blue', 'green']

    for i in range(3):
        plt.figure(figsize=(10, 6))
        plt.scatter(df[f'y_meas_{i}'], df[f'y_pred_{i}'], color=colors[i], alpha=0.5)
        plt.title(f'Actual vs. Predicted Values for Output {i}\nR^2 Score: {df[f"R2_{i}"].iloc[0]:.4f}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.axis('equal')
        plt.plot([df[f'y_meas_{i}'].min(), df[f'y_meas_{i}'].max()], [df[f'y_meas_{i}'].min(), df[f'y_meas_{i}'].max()], 'k--')  # Diagonal line
        plt.savefig(f'output_{i}_scatter_plot_other.png')  # Save plot to file





if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_time=end_time-start_time
    print(f"The total running time is {total_time}s")
    print(f"The total running time is {total_time/60}mins")