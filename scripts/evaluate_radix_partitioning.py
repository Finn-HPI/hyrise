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
time_data.plot(kind="bar", stacked=True, color=["blue", "yellow", "green"], figsize=(10, 6))

# Enhance the plot
plt.xlabel("Number of Partitions", fontsize=14)
plt.ylabel("Time (ms)", fontsize=14)
plt.title("Radix Partitioning Times", fontsize=16)
plt.legend(title="Phase", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()#
# # Create the plot
# plt.figure(figsize=(10, 6))
# bar_plot = sns.barplot(
#     data=melted_data,
#     x="num_partitions",
#     y="Time (ms)",
#     hue="Phase",
#     palette="muted"
# )
#
# # Enhance the plot
# bar_plot.set_title("Stacked Bar Plot of Partitioning Times", fontsize=16)
# bar_plot.set_xlabel("Number of Partitions", fontsize=14)
# bar_plot.set_ylabel("Time (ms)", fontsize=14)
# bar_plot.legend(title="Phase", fontsize=12)
#
# # Rotate x-axis labels for better readability
# bar_plot.set_xticklabels(bar_plot.get_xticklabels(), rotation=45, horizontalalignment='right')
#
# # Show the plot
# plt.tight_layout()
# plt.show()
#
