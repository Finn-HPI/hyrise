import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Check for command line arguments
if len(sys.argv) < 4:
    print("Usage: python script.py <path_to_csv1> <path_to_csv2> <fixed_leaf_size>")
    sys.exit(1)

# Load data from both CSVs without a header row
csv_path1 = sys.argv[1]
csv_path2 = sys.argv[2]

# Read the fixed_leaf_size argument
fixed_leaf_size = int(sys.argv[3])

data1 = pd.read_csv(csv_path1, header=None)
data2 = pd.read_csv(csv_path2, header=None)

# Set column names
data1.columns = ['leaf_count', 'leaf_size', 'execution_time']
data2.columns = ['leaf_count', 'leaf_size', 'execution_time']

# Convert execution_time from nanoseconds to microseconds
data1['execution_time'] = data1['execution_time'] / 1000
data2['execution_time'] = data2['execution_time'] / 1000

# Extract columns for first file
leaf_count1 = data1['leaf_count']
leaf_size1 = data1['leaf_size']
execution_time1 = data1['execution_time']

# Extract columns for second file
leaf_count2 = data2['leaf_count']
leaf_size2 = data2['leaf_size']
execution_time2 = data2['execution_time']

# Filter data for the fixed leaf size
data1_fixed = data1[data1['leaf_size'] == fixed_leaf_size]
data2_fixed = data2[data2['leaf_size'] == fixed_leaf_size]

# Extract leaf_count and execution_time for the fixed leaf size
leaf_count1_fixed = data1_fixed['leaf_count']
execution_time1_fixed = data1_fixed['execution_time']

leaf_count2_fixed = data2_fixed['leaf_count']
execution_time2_fixed = data2_fixed['execution_time']

# Plot the comparison
plt.figure(figsize=(10, 6))

# Plot for file 1
plt.plot(leaf_count1_fixed, execution_time1_fixed, label=f'MultiwayMerging (leaf_size={fixed_leaf_size})', color='blue', marker='o')

# Plot for file 2
plt.plot(leaf_count2_fixed, execution_time2_fixed, label=f'KWayMerging (leaf_size={fixed_leaf_size})', color='green', marker='x')

# Labels and title
plt.xlabel('Leaf Count')
plt.ylabel('Execution Time (μs)')
plt.title(f'Execution Time Comparison for Fixed Leaf Size ({fixed_leaf_size})')

# Show legend
plt.legend()
plt.grid()

# Save the plot to a file
plt.savefig(f'fixed_leaf_size_{fixed_leaf_size}.png')

# Display the plot
plt.show()
