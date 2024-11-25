import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Plot stacked bar chart from CSV.")
parser.add_argument("csv_file", type=str, help="Path to the input CSV file.")
args = parser.parse_args()

# Read the CSV file
data = pd.read_csv(args.csv_file)

# Ensure num_partitions is not included in the bar stack
data.set_index("num_partitions", inplace=True)  # Set num_partitions as the index
time_data = data[["time_histogram", "time_init", "time_partition"]]  # Select only the times

# Plot the data
sns.set_theme(style="whitegrid")
time_data.plot(kind="bar", stacked=True, color=["#a00000", "#1a80bb", "#384860" ], figsize=(10, 6))

# Enhance the plot
plt.xlabel("Number of Partitions", fontsize=14)
plt.ylabel("Time (ms)", fontsize=14)
plt.title("Radix Partitioning: 100 million entries ", fontsize=16)
plt.legend(title="Phase", labels=["Construct Histogram", "Initialize Buffer", "Partitioning"], fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(args.csv_file + '.png')

# Show the plot
plt.show()#

