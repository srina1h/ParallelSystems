import numpy as np
import matplotlib.pyplot as plt

# Input your data here
# Each row corresponds to a matrix, and each column corresponds to a runtime for a specific run
runtimes = [
    [0.007418, 0.0045053, 0.007589, 0.023059, 0.016146],
    [0.014374, 0.0139595, 0.014738, 0.036330, 0.033297],
    [0.010733, 0.0056646, 0.007728, 0.021429, 0.021879],
    [0.005181, 0.0005743, 0.003061, 0.012574, 0.007168],
    [0.004103, 0.0005659, 0.004526, 0.011809, 0.007810],
    [0.002600, 0.0005755, 0.002633, 0.011887, 0.007848]
]

# Define matrix labels (modify as needed)
matrix_labels = ['D6-6', 'dictionary28', 'Ga3As3H12', 'bfly', 'pkustk14', 'roadNet-CA']
matrix_labels.reverse()

# Number of bars per matrix
num_bars_per_matrix = len(runtimes[0])

# Prepare data for plotting
bar_width = 0.15
x = np.arange(len(matrix_labels))  # the label locations

bar_labels = ['MPI', 'OpenMP', 'hyb-8-8', 'hyb-4-8', 'hyb-2-8']

# Create a bar graph
fig, ax = plt.subplots(figsize=(12, 6))
for i in range(num_bars_per_matrix):
    ax.bar(x + i * bar_width, [row[i] for row in runtimes], width=bar_width, label=bar_labels[i])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Matrices')
ax.set_ylabel('Runtime (s)')
ax.set_title('Runtime of Different Matrices using different parallelization strategies')
ax.set_xticks(x + bar_width * (num_bars_per_matrix - 1) / 2)
ax.set_xticklabels(matrix_labels)
ax.legend()

# Show the plot
plt.tight_layout()
plt.savefig('runtime_graph.png')