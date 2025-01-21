##############################################################################
### Raw data visualization and normalization for preparation of NN training
### Data from LLC converter simulation on PSIM for Zuo's LLC modeling paper
### Author: Fanghao Tian
### Date: 2024-April-3rd
##############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rawdata = pd.read_csv('large_data_raw.csv')


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
number_same = same_data.shape[0]
other_data = rawdata[(rawdata.loc[:, 'dA'] != rawdata.loc[:, 'dB']) | (rawdata.loc[:, 'pA'] != rawdata.loc[:, 'pB'])]



#take out 10% from other data and same_data, and save them separately as test data same and normal, and then combine as test data all
test_same = same_data.sample(frac=0.1)
delete_same = same_data.drop(test_same.index)
test_other = other_data.sample(frac=0.1)
delete_other = other_data.drop(test_other.index)
test_data = pd.concat([test_same, test_other])
test_data.to_csv('test_data_both.csv',index=False)
print(test_data.shape[0])
#delete the test data from rawdata
rawdata = pd.concat([delete_same, delete_other])
print(rawdata.shape[0])


min_values = rawdata.min()
max_values = rawdata.max()

boundary_df = pd.DataFrame([min_values, max_values], index=['min', 'max']).transpose()
boundary_df.to_csv('boundary_large.csv', index_label='Column')

normalized_df = pd.DataFrame(columns=rawdata.columns)

n_cols = len(rawdata.columns)
n_rows = int(np.ceil(n_cols / 2))  # Adjust the denominator to change layout
fig, axes = plt.subplots(n_rows, 2, figsize=(15, n_rows * 5))
axes = axes.flatten()  # Flatten in case of a single row

# Iterate through each column and its corresponding subplot
for idx, column in enumerate(rawdata.columns):
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

# Adjust layout
plt.tight_layout()
normalized_df.to_csv("normalized_large.csv",index=False)
# Save the figure
plt.savefig("distribution_large.png")

# Normalize test_data
normalized_test = (test_data - rawdata.min()) / (rawdata.max() - rawdata.min())
normalized_test.to_csv("normalized_test.csv", index=False)

# Normalize test_same
normalized_test_same = (test_same - rawdata.min()) / (rawdata.max() - rawdata.min())
normalized_test_same.to_csv("normalized_test_same.csv", index=False)

# Normalize test_other
normalized_test_other = (test_other - rawdata.min()) / (rawdata.max() - rawdata.min())
normalized_test_other.to_csv("normalized_test_other.csv", index=False)




