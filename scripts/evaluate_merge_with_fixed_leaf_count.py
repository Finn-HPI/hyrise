import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Check for command line arguments
if len(sys.argv) < 4:
    print("Usage: python script.py <path_to_csv1> <path_to_csv2> <fixed_leaf_count>")
    sys.exit(1)

# Load data from both CSVs without a header row
csv_path1 = sys.argv[1]
csv_path2 = sys.argv[2]

# Read the leaf_count argument
fixed_leaf_count = int(sys.argv[3])

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

# Filter data for the fixed leaf count
data1_fixed = data1[data1['leaf_count'] == fixed_leaf_count]
data2_fixed = data2[data2['leaf_count'] == fixed_leaf_count]

# Extract leaf_size and execution_time for the fixed leaf count
leaf_size1_fixed = data1_fixed['leaf_size']
execution_time1_fixed = data1_fixed['execution_time']

leaf_size2_fixed = data2_fixed['leaf_size']
execution_time2_fixed = data2_fixed['execution_time']

# Plot the comparison
plt.figure(figsize=(10, 6))

# Plot for file 1
plt.plot(leaf_size1_fixed, execution_time1_fixed, label=f'MultiwayMerging (leaf_count={fixed_leaf_count})', color='blue', marker='o')

# Plot for file 2
plt.plot(leaf_size2_fixed, execution_time2_fixed, label=f'KWayMerging (leaf_count={fixed_leaf_count})', color='red', marker='x')

# Labels and title
plt.xlabel('Leaf Size')
plt.ylabel('Execution Time (μs)')
plt.title(f'Execution Time Comparison for Fixed Leaf Count ({fixed_leaf_count})')

# Show legend
plt.legend()
plt.grid()

plt.savefig(f'fixed_leaf_count_{fixed_leaf_count}.png')
# Display the plot
plt.show()
