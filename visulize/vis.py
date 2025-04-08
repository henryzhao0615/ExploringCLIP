import os
import matplotlib.pyplot as plt

# Ensure the output directory exists
output_dir = 'fig'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the shot labels and the corresponding metric values
shots_labels = ['0', '1', '2', '4', '8', '16', 'Full']

# Data from the table:
metrics = {
    'Top-1 Accuracy': [86.04, 74.84, 78.37, 82.54, 87.39, 90.23, 94.51],
    'Top-5 Accuracy': [99.17, 97.73, 98.17, 99.01, 98.75, 99.61, 99.87],
    'Precision':     [87.07, 77.35, 84.53, 83.23, 88.54, 90.56, 94.55],
    'Recall':        [86.04, 74.84, 78.37, 82.54, 87.39, 90.23, 94.51],
    'F1-score':      [85.96, 74.75, 79.55, 82.66, 87.50, 90.25, 94.52]
}

# Create numeric positions for the categorical x-axis
x = list(range(len(shots_labels)))

# Use a publication-quality style
plt.style.use('seaborn-whitegrid')

# Create subplots arranged in a 2x3 grid; the last subplot will be removed
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()  # Flatten for easy indexing

# Loop through each metric and create a subplot
for i, (metric_name, values) in enumerate(metrics.items()):
    ax = axes[i]
    
    # Plot bar chart for each metric
    bars = ax.bar(x, values, color='lightgray', edgecolor='black', zorder=2)
    
    # Overlay a line plot with markers connecting the bar centers
    ax.plot(x, values, marker='o', linestyle='-', linewidth=2, color='black', zorder=3)
    
    # Highlight the zero-shot point (first point) with a star marker
    ax.plot(x[0], values[0], marker='*', markersize=15, linestyle='None', color='red', zorder=4, label='Zero-shot')
    
    # Annotate each bar with its value
    for xi, val in zip(x, values):
        ax.text(xi, val + 0.5, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
    
    # Set labels and title for the subplot
    ax.set_xlabel('Number of Shots', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(metric_name, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(shots_labels, fontsize=12)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    
    # Add legend only on the first subplot to avoid clutter
    if i == 0:
        ax.legend(fontsize=12, loc=0)

# Remove the unused subplot (last one in the 2x3 grid)
if len(metrics) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()

# Save the multi-panel figure in high resolution
plt.savefig(os.path.join(output_dir, 'all_metrics.png'), bbox_inches='tight', dpi=300)
plt.show()
