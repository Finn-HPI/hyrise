import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import sys

# Check for command line arguments
if len(sys.argv) < 3:
    print("Usage: python script.py <path_to_csv1> <path_to_csv2>")
    sys.exit(1)

# Load data from both CSVs without a header row
csv_path1 = sys.argv[1]
csv_path2 = sys.argv[2]

data1 = pd.read_csv(csv_path1, header=None)
data2 = pd.read_csv(csv_path2, header=None)

# Set column names
data1.columns = ["leaf_count", "leaf_size", "execution_time"]
data2.columns = ["leaf_count", "leaf_size", "execution_time"]

# Extract columns for first file
leaf_count1 = data1["leaf_count"]
leaf_size1 = data1["leaf_size"]
execution_time1 = data1["execution_time"] / 1000

# Extract columns for second file
leaf_count2 = data2["leaf_count"]
leaf_size2 = data2["leaf_size"]
execution_time2 = data2["execution_time"] / 1000

# Create meshgrids for both datasets
leaf_count_unique1 = np.unique(leaf_count1)
leaf_size_unique1 = np.unique(leaf_size1)
leaf_count_grid1, leaf_size_grid1 = np.meshgrid(leaf_count_unique1, leaf_size_unique1)
execution_time_grid1 = execution_time1.values.reshape(len(leaf_size_unique1), len(leaf_count_unique1))

leaf_count_unique2 = np.unique(leaf_count2)
leaf_size_unique2 = np.unique(leaf_size2)
leaf_count_grid2, leaf_size_grid2 = np.meshgrid(leaf_count_unique2, leaf_size_unique2)
execution_time_grid2 = execution_time2.values.reshape(len(leaf_size_unique2), len(leaf_count_unique2))

# Normalize the execution times for both datasets (same range)
norm = Normalize(
    vmin=min(execution_time1.min(), execution_time2.min()), vmax=max(execution_time1.max(), execution_time2.max())
)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the first surface with the 'Mako' colormap
surface1 = ax.plot_surface(leaf_count_grid1, leaf_size_grid1, execution_time_grid1, color="blue", alpha=0.4, norm=norm)

# Plot the second surface with the 'Rocket' colormap
surface2 = ax.plot_surface(leaf_count_grid2, leaf_size_grid2, execution_time_grid2, color="green", alpha=0.4, norm=norm)

# Labels
ax.set_xlabel("Leaf Count")
ax.set_ylabel("Leaf Size")
ax.set_zlabel("Execution Time (μs)")

# Create a custom legend
legend_elements = [
    Line2D([0], [0], color="blue", lw=4, label="MultiwayMerging"),
    Line2D([0], [0], color="green", lw=4, label="KWayMerging"),
]

ax.legend(handles=legend_elements, loc="upper left")


plt.show()
