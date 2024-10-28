# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:31:11 2024

@author: gabri
"""

" ------------   IMPORT STATEMENTS  -------------- "
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import scipy.stats as stats
import json
from functions_mcts_truss_all   import percentage_optimal_states



" ---------  MCTS EXHAUSTIVE ----------- "

# Set the font properties globally
matplotlib.rc('font', family='Arial', size=16)
# colors = ['blue', 'orange', 'green', 'red','purple', 'brown']

examples = {'Ororbia_1': {'path':    '0.3_0.0_10_1000'},
            'Ororbia_2': {'path':    '0.3_0.0_10_1000'},
            'Ororbia_3': {'path':    '0.5_0.0_10_1000'},
            'Ororbia_4': {'path':    '0.3_0.0_10_5000'},
            'Ororbia_5': {'path':    '0.5_0.0_10_10000'},
            'Ororbia_7': {'path':    '0.3_0.0_10_10000'},
                                                             }


example_names = ['Case 1', 'Case 2', 'Case 3',  'Case 4', 'Case 5', 'Case 6']

min_nodes_list               = []
min_states_list               = []
min_values_list               = []

FEM_counter_list            = []
results_exhaustive_terminal_list = []


for example, values in examples.items():

    path_name       = values['path']
    
    with open(f'validation/{example}/{example}_min_maxDisplacement_{path_name}.json', 'r') as json_file:
        min_nodes = json.load(json_file)
    
    with open(f'validation/{example}/{example}_exhaustive_terminal_min_nodes.json', 'r') as json_file:
        min_node_exhaustive = json.load(json_file)
    
    with open(f'validation/{example}/{example}_FEM_counter_maxDisplacement_{path_name}.json', 'r') as json_file:
        FEM_counter = json.load(json_file)
        
    # Load the exhaustive terminal results
    results_exhaustive_terminal = np.load(f"validation/{example}/{example}_exhaustive_terminal.npy")
    
    min_nodes_list.append(min_nodes)
    min_states_list.append(min_node_exhaustive[0]['state'])
    min_values_list.append(min_node_exhaustive[0]['attribute_value'])
    FEM_counter_list.append(FEM_counter)
    results_exhaustive_terminal_list.append(results_exhaustive_terminal)


" -------------  G R A P H S ------------- "
# env.render0(min_nodes_list[5]['state'])


percentile_scores   = []
error_bars          = []
mean_values         = []

for i in range(len(min_nodes_list)):
    
    # Get attribute values for min_nodes
    attribute_values = [node['maxDisplacement'] for node in min_nodes_list[i]]
    
    # Calculate the mean and standard deviation of attribute values for error bars
    mean_attr_value = np.mean(attribute_values)
    std_attr_value = np.std(attribute_values)
    mean_values.append(mean_attr_value)
    error_bars.append(std_attr_value)
    
    # Calculate the percentile score for the min attribute value
    percentile_score = 100 - stats.percentileofscore(results_exhaustive_terminal_list[i], mean_attr_value)
    percentile_scores.append(percentile_score)
    
# Create the plot with error bars
plt.errorbar(example_names, percentile_scores, yerr=error_bars, fmt='o', label='Percentile Score with Std Dev', color='blue', ecolor='gray', elinewidth=3, capsize=5)

# Add labels and title
plt.xlabel('Example')
plt.ylabel('Percentile Score [%]')
# plt.title('Percentile Scores vs Examples')

# Add grid and legend
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.grid(axis='x', linestyle='-', alpha=0.7)
# plt.legend(loc='best')

# Display the plot
plt.show()



" GRAPH 1: Plotting the percentage of found optimal states by the algorithm"
percent_optimal_min_nodes     = percentage_optimal_states(min_nodes_list, min_states_list, 10)

plt.bar(example_names, percent_optimal_min_nodes,  color='skyblue')
plt.ylabel('Percentage of Optimal States found %')
plt.title('Percentage of optimal states found per example')
# Optionally, you can add the actual average above each bar
for i, avg in enumerate(percent_optimal_min_nodes):
    plt.text(i, avg, f'{avg:.2f}', ha='center', va='bottom')
    
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.grid(axis='x', linestyle='-', alpha=0.7)
plt.show()



" GRAPH 1: Plotting the percentage of found optimal states by the algorithm"

percent_optimal_min_nodes     = percentage_optimal_states(min_nodes_list, min_states_list, 10)

second_list_values = [100, 100, 100, 60, 100, 60]  # Second set of values to plot

# Number of groups
n_groups = len(example_names)

# Create figure and axes
fig, ax = plt.subplots()

# Set the positions of the bars on the x-axis
index = np.arange(n_groups)
bar_width = 0.35

# Plot the first set of values
rects1 = ax.bar(index - bar_width/2, percent_optimal_min_nodes, bar_width, label='MCTS algorithm', color='skyblue')

# Plot the second set of values
rects2 = ax.bar(index + bar_width/2, second_list_values, bar_width, label='Ororbia Article', color='lightgreen')

# Add labels, title, and custom x-axis tick labels
ax.set_ylabel('Percentage %')
ax.set_title('Percentage of optimal states found per example')
ax.set_xticks(index)
ax.set_xticklabels(example_names)
ax.legend(loc='lower left')

# Optionally, add text for the percentage above each bar
for i in range(n_groups):
    ax.text(i - bar_width/2, percent_optimal_min_nodes[i], f'{percent_optimal_min_nodes[i]:.0f}', ha='center', va='bottom')
    ax.text(i + bar_width/2, second_list_values[i], f'{second_list_values[i]:.0f}', ha='center', va='bottom')

# Show the plot
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.show()


" GRAPH: Number of FEM simulations per example against ORORBIA"
# Calculate the average of each sublist
averages = [np.mean(sublist) for sublist in FEM_counter_list]

second_list_values = [419, 2179, 2221, 5735, 33284, 11536]  # Second set of values to plot

# Number of groups
n_groups = len(example_names)

# Create figure and axes
fig, ax = plt.subplots()

# Set the positions of the bars on the x-axis
index = np.arange(n_groups)
bar_width = 0.35

# Plot the first set of values
rects1 = ax.bar(index - bar_width/2, averages, bar_width, label='MCTS', color='skyblue')

# Plot the second set of values
rects2 = ax.bar(index + bar_width/2, second_list_values, bar_width, label='Deep Q-Learning', color='lightgreen')

# Add labels, title, and custom x-axis tick labels
ax.set_ylabel('# FE Evaluations')
ax.set_xticks(index)
ax.set_xticklabels(example_names, rotation=45)  # Rotate x-axis labels

ax.set_yticks(np.arange(0, 35000, 5000))

ax.legend(loc='upper left', fontsize='small')

# Optionally, add text for the percentage above each bar
# for i in range(n_groups):
#     ax.text(i - bar_width/2, averages[i], f'{averages[i]:.0f}', ha='center', va='bottom')
#     ax.text(i + bar_width/2, second_list_values[i], f'{second_list_values[i]:.0f}', ha='center', va='bottom')

# Show the plot
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.show()




" GRAPH: Number of FEM simulations per example"
# Plotting
plt.figure()  # Set the figure size as desired
plt.bar(example_names, averages, color='skyblue')  # Use a bar chart to display averages

# Adding labels and title
plt.ylabel('FEM Evaluations')
# plt.title('FEM simulations per example')

# Optionally, you can add the actual average above each bar
# for i, avg in enumerate(averages):
#     plt.text(i, avg, f'{avg:.0f}', ha='center', va='bottom')

# Show the plot
plt.tight_layout()  # Adjust the padding between and around subplots
plt.grid(axis='y', linestyle='-', alpha=0.7)
# plt.grid(axis='x', linestyle='-', alpha=0.7)
plt.show()



" HORIZONTAL GRAPH: Number of FEM simulations per example"
averages = [np.mean(sublist) for sublist in FEM_counter_list]

# Number of groups
n_groups = len(example_names)

# Create figure and axes
fig, ax = plt.subplots()

# Set the positions of the bars on the y-axis
index = np.arange(n_groups)
bar_width = 0.35

# Plot the first set of values
rects1 = ax.barh(index - bar_width/2, averages, bar_width, label='Proposed MCTS Framework', color='skyblue')

# Add labels, title, and custom y-axis tick labels
ax.set_xlabel('# FEM simulations')
# ax.set_title('FEM simulations per example')
ax.set_yticks(index)
ax.set_yticklabels(example_names)
ax.set_xticks(np.arange(0, 13000, 2500))  # Adjust as necessary

ax.legend(loc='upper right', fontsize='small')

# Optionally, add text for the values next to each bar
for i in range(n_groups):
    ax.text(averages[i] + 150, index[i] - bar_width/2, f'{averages[i]:.0f}', va='center', fontsize=14)

# Invert the y-axis to have the first item at the top
ax.invert_yaxis()

# Show the plot
plt.grid(axis='x', linestyle='-', alpha=0.7)
plt.show()




" HORIZONTAL GRAPH: Number of FEM simulations per example"
averages = [np.mean(sublist) for sublist in FEM_counter_list]
second_list_values = [419, 2179, 2221, 5735, 33284, 11536]

# Number of groups
n_groups = len(example_names)

# Create figure and axes
fig, ax = plt.subplots()

# Set the positions of the bars on the y-axis
index = np.arange(n_groups)
bar_width = 0.35

# Plot the first set of values
rects1 = ax.barh(index - bar_width/2, averages, bar_width, label='MCTS', color='skyblue')

# Plot the second set of values
rects2 = ax.barh(index + bar_width/2, second_list_values, bar_width, label='Deep Q-Learning', color='lightgreen')

# Add labels, title, and custom y-axis tick labels
ax.set_xlabel('# FE evaluations')
# ax.set_title('FEM simulations per example')
ax.set_yticks(index)
ax.set_yticklabels(example_names)
ax.set_xticks(np.arange(0, 41000, 10000))  # Adjust as necessary

ax.legend(loc='upper right', fontsize='small')

# Optionally, add text for the values next to each bar
for i in range(n_groups):
    ax.text(averages[i] + 1000, index[i] - bar_width/2, f'{averages[i]:.0f}', va='center', fontsize=12)
    ax.text(second_list_values[i] + 1000, index[i] + bar_width/2, f'{second_list_values[i]:.0f}', va='center', fontsize=12)

# Invert the y-axis to have the first item at the top
ax.invert_yaxis()

# Show the plot
plt.grid(axis='x', linestyle='-', alpha=0.7)
plt.show()




