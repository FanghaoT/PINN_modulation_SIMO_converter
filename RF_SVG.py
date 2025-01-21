import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os
from sklearn.svm import SVR

def train_and_evaluate(method="RF",dataset='small'):
    """
    Train and evaluate the model using the specified method.

    Parameters:
        method (str): Regression method, "RF" for Random Forest or "SVR" for Support Vector Regression.
    """
    rawdata_train = pd.read_csv(os.path.join(f'normalized_{dataset}.csv'))
    
    # Features and target variable for training
    X_train = rawdata_train[['pA', 'pB', 'dA', 'dB', 'dP']]
    y_train = rawdata_train[['IB', 'IA', 'IP']]

    # Load test data
    rawdata_test = pd.read_csv(os.path.join('normalized_test.csv'))

    # Features and target variable for testing
    X_test = rawdata_test[['pA', 'pB', 'dA', 'dB', 'dP']]
    y_test = rawdata_test[['IB', 'IA', 'IP']]

    # Choose regression method
    models = []
    if method == "RF":
        # Random Forest Regressor
        for i in range(y_train.shape[1]):
            model = RandomForestRegressor(n_estimators=10, random_state=36)
            model.fit(X_train, y_train.iloc[:, i])
            models.append(model)
    elif method == "SVR":
        # Support Vector Regression
        for i in range(y_train.shape[1]):
            model = SVR(kernel='rbf', C=0.5, epsilon=0.1)
            model.fit(X_train, y_train.iloc[:, i])
            models.append(model)

    # Predictions
    y_pred = np.column_stack([model.predict(X_test) for model in models])


    inputs_array = X_test.values
    y_meas = y_test.values

    # Load normalization boundaries
    boundaries = pd.read_csv(f"boundary_{dataset}.csv")
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

# Example usage
# Choose 'RF' for Random Forest or 'SVR' for Support Vector Regression
train_and_evaluate(method="RF",dataset='small')
train_and_evaluate(method="RF",dataset='large')
train_and_evaluate(method="SVR",dataset='small')
train_and_evaluate(method="SVR",dataset='large')