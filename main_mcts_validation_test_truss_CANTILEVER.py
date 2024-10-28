" ------------   IMPORT STATEMENTS  -------------- "
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import scipy.stats as stats
import json
# from functions_MCTS_print       import histogram_plot
from functions_mcts_truss_all   import percentage_optimal_states #, calculate_means_stds


" ------------  TRUSS -------------- "
from env_truss                  import env_truss

example                 = "cantilever"
env                     = env_truss(example)
attr_to_minimize        = 'maxDisplacement'


def calculate_means_stds(sublists):
    means = np.mean(sublists, axis=0)
    stds = np.std(sublists, axis=0)
    return means, stds



" ---------  MCTS VALIDATION RUNS ----------- "
# alpha_values            = [0.5]
alpha_values            = [0.1,0.3, 0.5, 0.7]
beta_values             = [0.0]
total_simulations       = 10
episodes_length         = 1000

min_nodes_list               = []
elapsed_times           = []
convergence_points           = []
FEM_counter_list           = []


for alpha in alpha_values:
    for beta in beta_values:
        min_path_name       = f"{example}_min_{attr_to_minimize}_{alpha}_{beta}_{total_simulations}_{episodes_length}"
        time_path_name      = f"{example}_time_{attr_to_minimize}_{alpha}_{beta}_{total_simulations}_{episodes_length}"
        convergence_path_name      = f"{example}_convergence_{attr_to_minimize}_{alpha}_{beta}_{total_simulations}_{episodes_length}"
        FEM_path_name      = f"{example}_FEM_counter_{attr_to_minimize}_{alpha}_{beta}_{total_simulations}_{episodes_length}"
        
        with open(f'validation/{example}/{min_path_name}.json', 'r') as json_file:
            min_data = json.load(json_file)
        
        with open(f'validation/{example}/{time_path_name}.json', 'r') as json_file:
            time_data = json.load(json_file)
        
        with open(f'validation/{example}/{convergence_path_name}.json', 'r') as json_file:
            convergence_data = json.load(json_file)
        
        with open(f'validation/{example}/{FEM_path_name}.json', 'r') as json_file:
            FEM_data = json.load(json_file)
            
        min_nodes_list.append(min_data)
        elapsed_times.append(time_data)
        convergence_points.append(convergence_data)
        FEM_counter_list.append(FEM_data)





" -------------  G R A P H S ------------- "
# for min_node in min_nodes_list[0]:
#     env.render0_black(min_node['state'], r'$s_{6}$')
#     print(min_node['maxDisplacement'])
# 
# env.render0_black(env.initial_state, r'$s_{0}$')




# Set the font properties globally
matplotlib.rc('font', family='Arial', size=24)
colors = ['blue', 'orange', 'green', 'red','purple', 'brown']



# Find the min value of maxDisplacement
min_state_exhaust = min_nodes_list[0][0]['state']
min_value_exhaust = min_nodes_list[0][0]['maxDisplacement']


print(f"Min maxDisplacement value: {min_value_exhaust}")
print(f"State associated with min maxDisplacement: {min_state_exhaust}")
env.render0(min_state_exhaust, "Global Minimum configuration")
env.render0_black(min_state_exhaust, r'$s_{3}$')






" GRAPH 1: Plotting the percentage of found optimal states by the algorithm"
percental_optimal_min_nodes     = percentage_optimal_states(min_nodes_list, [min_state_exhaust], total_simulations)
# Convert the line plot to a column graph
plt.figure()
bar_width = 0.1  # Width of the bars
positions = np.arange(len(alpha_values))  # Position of bars on x-axis

plt.bar(positions, percental_optimal_min_nodes, width=bar_width)

# Adding labels, title, and custom x-axis tick labels
plt.xlabel('Alpha')
plt.ylabel('Percentage of Optimal States found %')
plt.title('Percentage of optimal states found vs Alpha')
plt.xticks(positions, [str(alpha) for alpha in alpha_values])

# Optionally, add the percentage above each bar
for i, percentage in enumerate(percental_optimal_min_nodes):
    plt.text(i, percentage + 1, f'{percentage}%', ha='center', va='bottom')


y_max = max(percental_optimal_min_nodes) + 5  # Extend 10 units below the minimum score, or to 0

plt.ylim([0, y_max])
plt.grid(axis='y', linestyle='-', alpha=0.7)
# plt.legend()
plt.show()




" GRAPH: Number of FEM simulations per example"
# Calculate the average of each sublist
averages = [np.mean(sublist) for sublist in FEM_counter_list]

# averages = [998]
# Plotting
plt.figure(figsize=(10, 6))
bar_width=0.05
plt.bar(alpha_values, averages, width=bar_width,color='skyblue')  # Use a bar chart to display averages

# Adding labels and title
plt.xlabel('$\\alpha$')
plt.ylabel('FEM Evaluations')
# plt.title('FEM simulations vs Alpha')
plt.xticks(alpha_values, [str(alpha) for alpha in alpha_values])

# Optionally, you can add the actual average above each bar
# for i, avg in enumerate(averages):
#     plt.text(alpha_values[i], avg, f'{avg:.0f}', ha='center', va='bottom')

# Show the plot
# plt.tight_layout()  # Adjust the padding between and around subplots
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.grid(axis='x', linestyle='-', alpha=0.7)
plt.show()






" GRAPH 3: Plotting the elapsed times vs alpha "
time_means = [np.mean(sublist) for sublist in elapsed_times]
time_stds = [np.std(sublist) for sublist in elapsed_times]

# Plotting mean elapsed times with standard deviation error bars
plt.errorbar(alpha_values, time_means, yerr=time_stds, fmt='o', label='Elapsed Time with Std Dev', color='blue', ecolor='gray', elinewidth=3, capsize=5)

plt.xlabel('Alpha')
plt.ylabel('Elapsed Time [s]')
plt.title('Elapsed Time vs Alpha')
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.grid(axis='x', linestyle='-', alpha=0.7)
plt.xticks(alpha_values, [str(alpha) for alpha in alpha_values])

# plt.legend()
plt.show()



" GRAPH 4: Plotting the simulation convergence episodes vs alpha "

# Iterate over each set of sublists
for index, sublists in enumerate(convergence_points):
    plt.figure(figsize=(10, 6))
    means, stds = calculate_means_stds(sublists)
    x_values = list(range(len(sublists[0])))
        
    plt.plot(x_values, means, label=f'$\\alpha$: {alpha_values[index]:.2f}',  linewidth=3)
    plt.fill_between(x_values, np.subtract(means, stds), np.add(means, stds), alpha=0.2)

    # # Add a horizontal line for the global minimum attribute value
    plt.axhline(y=min_value_exhaust, color='red', linestyle='dashed', linewidth=3)
    # plt.text(max(x_values), min_value_exhaust, f'Global Minimum: {min_value_exhaust:.4f} (100%)', va='top', ha='right', color='red')
        
    
    plt.xlabel('Episode')
    plt.ylabel('Displacement')
    # plt.title('Min Displacement Result vs Episodes')
    plt.legend(loc='upper right', fontsize='20')
    # plt.grid(axis='y', linestyle='-', alpha=0.7)
    # plt.grid(axis='x', linestyle='-', alpha=0.7)
    
    
    plt.yticks(np.arange(20, 35, 4))
    plt.xlim([0, 300])
    plt.ylim([20, 36])


    plt.show()


















