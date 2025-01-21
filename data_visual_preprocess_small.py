##############################################################################
### Raw data visualization and normalization for preparation of NN training
### Data from LLC converter simulation on PSIM for Zuo's LLC modeling paper
### Author: Fanghao Tian
### Date: 2024-April-3rd
##############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rawdata = pd.read_csv('small_data_diego_and_new.csv')
# rawdata = rawdata[(rawdata.loc[:, 'TRUE'] == 1)]     #只选取你标记了1的数据
# rawdata.to_csv('onlytrue.csv')

#print the length of the data
print(rawdata.shape[0])

#clean the data
#1.delete the data with dA,dB or dP <0.1
rawdata = rawdata[(rawdata.loc[:, 'dA'] >= 0.1) & (rawdata.loc[:, 'dB'] >= 0.1) & (rawdata.loc[:, 'dP'] >= 0.1)]
#2.delete the data with IP <0.1
rawdata = rawdata[(abs(rawdata.loc[:, 'IP']) >= 0.1)]
#3.delete the data with abs（IA）<1 and abs（IB）<1
rawdata = rawdata[(abs(rawdata.loc[:, 'IA']) >= 1) & (abs(rawdata.loc[:, 'IB']) >= 1)]
print(rawdata.shape[0])

#Analyze the rest of the data, some of them are dA==dB and pA==pB, seperate them out and print the size
same_data = rawdata[(rawdata.loc[:, 'dA'] == rawdata.loc[:, 'dB']) & (rawdata.loc[:, 'pA'] == rawdata.loc[:, 'pB'])]
print(same_data.shape[0])
# number_same = same_data.shape[0]
# other_data = rawdata[(rawdata.loc[:, 'dA'] != rawdata.loc[:, 'dB']) | (rawdata.loc[:, 'pA'] != rawdata.loc[:, 'pB'])]
#Now pick up random number_same data from other_data, and combine them with same_data, forme the new dataset called rawdata
#and how do I ensure all the min and max values are included in the number_same data?
# other_data = other_data.sample(n=number_same)

# rawdata = rawdata[(rawdata.loc[:, 'TRUE'] == 1)]     #只选取你标记了1的数据
# rawdata.to_csv('onlytrue.csv')
min_values = rawdata.min()
max_values = rawdata.max()

boundary_df = pd.DataFrame([min_values, max_values], index=['min', 'max']).transpose()
boundary_df.to_csv('boundary_small.csv', index_label='Column')

normalized_df = pd.DataFrame(columns=rawdata.columns)

n_cols = len(rawdata.columns)
n_rows = int(np.ceil(n_cols / 2))  # Adjust the denominator to change layout
fig, axes = plt.subplots(n_rows, 2, figsize=(15, n_rows * 5))
axes = axes.flatten()  # Flatten in case of a single row

# Iterate through each column and its corresponding subplot
for idx, column in enumerate(rawdata.columns):
    #对部分分布不均的数据（大量数据集中在小值的情况）取log10
    # if column == 'IB' or column == 'IA':
    #     rawdata[column] = rawdata[column].apply(lambda x: max(x, 1) if x >= 0 else min(x, -1))
    # if column == 'IP':
    #     rawdata[column] = rawdata[column].apply(lambda x: max(x, 0.1) if x >= 0 else min(x, -0.1))

    # Normalize the data to [0, 1]
    normalized_data = (rawdata[column] - rawdata[column].min()) / (rawdata[column].max() - rawdata[column].min())
    # normalized_data = rawdata[column]
    normalized_df[column] = normalized_data  # Store the normalized data
    # Bin the data
    bins = np.arange(0, 1.03, 0.03)  # Adjust the step size if necessary
    binned_data = pd.cut(normalized_data, bins, include_lowest=True, right=False)
    
    # Count the number of data points in each bin
    bin_counts = binned_data.value_counts(sort=False)
    
    # Plot the distribution
    bin_counts.plot(kind='bar', ax=axes[idx])
    axes[idx].set_title(f'Distribution for {column}')
    axes[idx].set_xlabel('Normalized Data Interval')
    axes[idx].set_ylabel('Count')
    axes[idx].tick_params(axis='x', rotation=45)

print(normalized_df.shape[0])
# Load the test dataset
normalized_test = pd.read_csv('normalized_test.csv')
print(f"Test dataset size: {normalized_test.shape[0]}")
# Log initial sizes
print(f"Initial size of normalized_df: {normalized_df.shape[0]}")
print(f"Size of normalized_test: {normalized_test.shape[0]}")

# Step 1: Ensure consistent data types
normalized_test = normalized_test.astype(normalized_df.dtypes)

# Step 2: Reset indices for accurate comparison
normalized_df = normalized_df.reset_index(drop=True)
normalized_test = normalized_test.reset_index(drop=True)

# Step 3: Handle floating-point precision (rounding numeric columns)
numeric_columns = normalized_df.select_dtypes(include=['float64', 'float32']).columns
normalized_df[numeric_columns] = normalized_df[numeric_columns].round(8)
normalized_test[numeric_columns] = normalized_test[numeric_columns].round(8)

# Step 4: Concatenate and drop duplicates
combined_df = pd.concat([normalized_df, normalized_test], ignore_index=True)
normalized_df = combined_df.drop_duplicates(keep=False)

# Explicitly drop rows that were in `normalized_test`
normalized_df = normalized_df[~normalized_df.apply(tuple, axis=1).isin(normalized_test.apply(tuple, axis=1))]

# Log final size
print(f"Final size of normalized_df (after dropping duplicates and test points): {normalized_df.shape[0]}")

# print(f"Filtered normalized_df size: {normalized_df.shape[0]}")
# print(normalized_df.shape[0])

# # Load the test dataset
# normalized_test = pd.read_csv('normalized_test.csv')
# print(normalized_test.shape[0])

# # Concatenate the dataframes
# combined_df = pd.concat([normalized_df, normalized_test])

# print(combined_df.shape[0])

# # Remove duplicates, keeping only rows that are unique to `normalized_df`
# normalized_df = combined_df.drop_duplicates(keep=False)

# print(normalized_df.shape[0])

# Adjust layout
plt.tight_layout()
normalized_df.to_csv("normalized_small.csv",index=False)
# Save the figure
plt.savefig("distribution_small.png")






