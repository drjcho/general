import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
# file_path = './HOPE-Data-Kor-Master-data-only-students.csv'
file_path = './HOPE-Data-Kor-Master-data.csv'

data = pd.read_csv(file_path)

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix using a heatmap
# %%
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='binary', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Display correlation matrix as a table
#import ace_tools as tools
#tools.display_dataframe_to_user(name="Correlation Matrix", dataframe=correlation_matrix)


# %%
# Nomalize the AveAca and AveSocio (remove differences)
# Specify columns to normalize
columns_to_normalize = ['AveAca', 'AveSocio']

# Calculate and store the standard deviation before normalization
data['AveAca_StdDev_Before'] = data.groupby('Teacher ID')['AveAca'].transform('std')
data['AveSocio_StdDev_Before'] = data.groupby('Teacher ID')['AveSocio'].transform('std')

# Normalize the specified columns based on Teacher ID
data[columns_to_normalize] = data.groupby('Teacher ID')[columns_to_normalize].transform(lambda x: (x - x.mean()) / x.std())

# Calculate and store the standard deviation after normalization
data['AveAca_StdDev_After'] = data.groupby('Teacher ID')['AveAca'].transform('std')
data['AveSocio_StdDev_After'] = data.groupby('Teacher ID')['AveSocio'].transform('std')

# Save the normalized data with standard deviations to a CSV file
data.to_csv('normalized_data_with_stddev.csv', index=False)
print("Normalized data with standard deviations saved to 'normalized_data_with_stddev.csv'")

# %%
file_path = './normalized_data_with_stddev_revised01.csv'

data2 = pd.read_csv(file_path)

# Calculate the correlation matrix
correlation_matrix = data2.corr()

# Plot the correlation matrix using a heatmap

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='binary', linewidths=0.5)
plt.title('Correlation Matrix Heatmap for Nomalized Scores')
plt.show()
# %%
