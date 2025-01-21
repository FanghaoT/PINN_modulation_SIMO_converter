
# Import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define a fully connected layers model with three inputs (frequency, flux density, duty ratio) and one output (power loss).
        self.layers = nn.Sequential(
            nn.Linear(5, 111),
            nn.ReLU(),
            nn.Linear(111, 184),
            nn.ReLU(),
            nn.Linear(184, 227),
            nn.ReLU(),
            nn.Linear(227, 216),
            nn.ReLU(),
            nn.Linear(216, 143),
            nn.ReLU(),
            nn.Linear(143, 3),
        )

    def forward(self, x):
        return self.layers(x)
  
# Load the trained model
model_path = 'Model_FNN.sd'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().double().to(device)
net.load_state_dict(torch.load(model_path, map_location=device))

# Set the model to evaluation mode
net.eval()

# Assuming the test dataset loader is set up as follows
def get_test_loader():
    # This function needs to be defined based on how you have your test data prepared.
    # Here is a simple placeholder:
    test_data = pd.read_csv('normalized.csv')
    # Assuming the data needs to be converted to tensors
    inputs = torch.tensor(test_data.iloc[:, :-3].values).double()
    labels = torch.tensor(test_data.iloc[:, -3:].values).double()
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    return loader

test_loader = get_test_loader()
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
boundaries = pd.read_csv("boundary.csv")
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
    plt.show()