import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset from a CSV file
# Replace 'iris.csv' with the path to your CSV file
iris_df = pd.read_csv('Iris.csv')

# Ensure the dataset has the correct column names
# Example: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']

# Display correlation matrix
correlation_matrix = iris_df.iloc[:, :-1].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Plot heatmap of the correlation matrix using matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')

# Add labels
plt.xticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns)

# Add title
plt.title('Correlation Matrix Heatmap')

# Annotate the heatmap with correlation values
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')

plt.tight_layout()
plt.show()

# Plot histograms for each numeric column
numeric_columns = iris_df.select_dtypes(include=[np.number]).columns
iris_df[numeric_columns].hist(bins=15, figsize=(12, 8), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numeric Features')
plt.tight_layout()
plt.show()

# Plot scatter matrix to visualize pairwise relationships
from pandas.plotting import scatter_matrix
scatter_matrix(iris_df.iloc[:, :-1], figsize=(12, 12), diagonal='hist', color='blue', alpha=0.7)
plt.suptitle('Scatter Matrix of Numeric Features')
plt.show()