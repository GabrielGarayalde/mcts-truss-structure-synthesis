" ------------   IMPORT STATEMENTS  -------------- "
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import scipy.stats as stats
import json
# from functions_MCTS_print       import histogram_plot
from functions_mcts_truss_all   import calculate_means_stds


" ------------  TRUSS -------------- "
from env_truss                  import env_truss

example                 = "Ororbia_4"
env                     = env_truss(example)
attr_to_minimize        = 'maxDisplacement'



" ---------  MCTS EXHAUSTIVE ----------- "

results_exhaustive_terminal = np.load(f"validation/{example}/{example}_exhaustive_terminal.npy")
min_value = min(results_exhaustive_terminal)

exhaustive_terminal_min_nodes_path          = f"validation/{example}/{example}_exhaustive_terminal_min_nodes"

with open(f'{exhaustive_terminal_min_nodes_path}.json', 'r') as json_file:
    min_node_exhaustive = json.load(json_file)

# Find the min value of maxDisplacement
min_value_exhaust = min_node_exhaustive[0]['attribute_value']
min_state_exhaust = min_node_exhaustive[0]['state']


print(f"Min maxDisplacement value: {min_value_exhaust}")
print(f"State associated with min maxDisplacement: {min_state_exhaust}")
# env.render0(min_state_exhaust, "Global Minimum configuration")
# env.render0_black(min_state_exhaust, r'$s_{3}$')



" ---------  MCTS VALIDATION RUNS ----------- "
# alpha_values            = [0.1, 0.2, 0.3, 0.4, 0.5]
alpha_values            = [0.5]
beta_values             = [0.0, 0.2, 0.4, 0.6, 0.8]
total_simulations       = 10
episodes_length         = 5000



min_nodes_list               = []
elapsed_times           = []
convergence_points           = []
FEM_counter_list           = []


for beta in beta_values:
    min_nodes_list_alpha               = []
    elapsed_times_alpha            = []
    convergence_points_alpha            = []
    FEM_counter_list_alpha            = []
    for alpha in alpha_values:
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
    
        min_nodes_list_alpha.append(min_data)
        elapsed_times_alpha.append(time_data)
        convergence_points_alpha.append(convergence_data)
        FEM_counter_list_alpha.append(FEM_data)
    
    min_nodes_list.append(min_nodes_list_alpha)
    elapsed_times.append(elapsed_times_alpha)
    convergence_points.append(convergence_points_alpha)
    FEM_counter_list.append(FEM_counter_list_alpha)



" -------------  G R A P H S ------------- "
# for min_node in min_nodes_list[1]:
#     env.render0(min_node['state'])
#     print(min_node['maxDisplacement'])






# Set the font properties globally
matplotlib.rc('font', family='Arial', size=16)

# Initialization
percentile_scores = [[] for _ in beta_values]  # Scores per beta
error_bars = [[] for _ in beta_values]  # Error bars per beta

for beta_index, beta_min_nodes in enumerate(min_nodes_list):
    for alpha_index, attribute_values in enumerate(beta_min_nodes):
        
        attribute_values = [node['maxDisplacement'] for node in attribute_values]

        # Calculate the mean and standard deviation of attribute values for error bars
        mean_attr_value = np.mean(attribute_values)
        std_attr_value = np.std(attribute_values)
        error_bars[beta_index].append(std_attr_value)
        
        # Calculate the percentile score for the min attribute value
        percentile_score = 100 - stats.percentileofscore(results_exhaustive_terminal, mean_attr_value)
        percentile_scores[beta_index].append(percentile_score)

# Plotting
plt.figure(figsize=(10, 6))

colors = ['blue', 'red']
for beta_index, beta in enumerate(beta_values):
    plt.errorbar(alpha_values, percentile_scores[beta_index], yerr=error_bars[beta_index], fmt='o-', label=f'Beta = {beta}', ecolor='gray', elinewidth=3, capsize=5)

# Add labels and title
plt.xlabel('Alpha values')
plt.ylabel('Percentile Score')
plt.title('Percentile Scores vs Alpha for different Beta values')

plt.xticks(alpha_values, [str(alpha) for alpha in alpha_values])

# Ensure the y-axis includes 100 and goes slightly below the lowest score
# y_min = min([min(scores) for scores in percentile_scores]) - 5
y_max = 100  # Ensure the y-axis goes up to 100
plt.ylim([98.5, 100])

# Add grid, legend, and display the plot
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.grid(axis='x', linestyle='-', alpha=0.7)
plt.legend(loc='best')
plt.show()



" GRAPH 1: Plotting the percentile of the min_nodes wrt to alpha"

# Plot setup
plt.figure(figsize=(10, 6))

for beta_index, beta in enumerate(beta_values):
    percentile_scores = []
    error_bars = []
    for alpha_index, alpha in enumerate(alpha_values):
        attribute_values = [node['maxDisplacement'] for node in min_nodes_list[beta_index][alpha_index]]
        
        # Calculate the mean and standard deviation
        mean_attr_value = np.mean(attribute_values)
        std_attr_value = np.std(attribute_values)
        
        # Calculate the percentile score
        percentile_score = 100 - stats.percentileofscore(results_exhaustive_terminal, mean_attr_value)
        percentile_scores.append(percentile_score)
        error_bars.append(std_attr_value)
        
    # Plotting the results for this beta value
    plt.plot(alpha_values, percentile_scores, '-o', label=f'Beta {beta}', alpha=0.7)
    plt.fill_between(alpha_values, np.array(percentile_scores) - np.array(error_bars), np.array(percentile_scores) + np.array(error_bars), alpha=0.2)

# Customizing the plot
plt.xlabel('Alpha')
plt.ylabel('Percentile Score')
plt.title('Percentile Scores vs Alpha for Different Beta Values')
plt.xticks(alpha_values, [str(alpha) for alpha in alpha_values])
plt.ylim([99, 100])
plt.grid(True)
plt.legend()
plt.show()






" GRAPH 1: Plotting the percentage of found optimal states by the algorithm"
# Adjusting the provided function for the new structure of `min_nodes_list` (list of lists for beta and alpha)

def percentage_optimal_states(chosen_nodes, optimal_states, total_simulations):
    def convert_state_to_set(state):
        "  Function to sort each sublist and convert to a tuple and then convert all sublists to a set "
        return set(tuple(sorted(sublist)) for sublist in state)

    optimal_state_sets = [convert_state_to_set(optimal_state) for optimal_state in optimal_states]
    percentage_optimal_per_beta_alpha = []  # Adjusted to accommodate beta dimension

    for chosen_nodes_beta in chosen_nodes:  # Iterate over beta values
        percentage_optimal_per_alpha = []  # Percentage for each alpha within current beta
        for chosen_nodes_alpha in chosen_nodes_beta:  # Iterate over alpha values
            count = 0
            for node in chosen_nodes_alpha:
                node_state_set = convert_state_to_set(node['state'])
                if any(node_state_set == optimal_state_set for optimal_state_set in optimal_state_sets):
                    count += 1
            percentage = (count / total_simulations) * 100  # Convert count to percentage
            percentage_optimal_per_alpha.append(percentage)
        percentage_optimal_per_beta_alpha.append(percentage_optimal_per_alpha)

    return percentage_optimal_per_beta_alpha

# # Calculate the percentage of optimal states found for each combination of alpha and beta
percental_optimal_min_nodes_beta_alpha = percentage_optimal_states(min_nodes_list, [min_state_exhaust], total_simulations)

plt.figure(figsize=(10, 6))

# Width of the bars
bar_width = 0.2

# Positions of the bars on the x-axis
r1 = np.arange(len(alpha_values))
r2 = [x + bar_width for x in r1]

for beta_index, beta in enumerate(beta_values):
    # Adjusting positions for each beta
    positions = [x + (beta_index * bar_width) for x in r1]
    
    bars = plt.bar(positions, percental_optimal_min_nodes_beta_alpha[beta_index], width=bar_width, label=f'Beta = {beta}')
    
    # Adding the percentage on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{int(yval)}%', ha='center', va='bottom')

# Adding labels, title and custom x-axis tick labels, etc.
plt.xlabel('Alpha')
plt.ylabel('Percentage of Optimal States found %')
plt.title('Percentage of optimal states found vs Alpha for different Beta values')
plt.xticks([r + bar_width for r in range(len(alpha_values))], [str(alpha) for alpha in alpha_values])

plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.legend()
plt.show()







" GRAPH: Number of FEM simulations per example"
# Assuming `FEM_counter_list` contains data structured similar to `percental_optimal_min_nodes_beta_alpha`:
# A list of lists (for each beta), each containing lists of FEM counter values (for each alpha).

# Calculate the average number of FEM simulations for each alpha within each beta
averages_beta_alpha = [
    [np.mean(sublist_alpha) for sublist_alpha in sublist_beta]
    for sublist_beta in FEM_counter_list
]

plt.figure(figsize=(10, 6))

# Adjust bar positions and widths for clarity in visualization
bar_width = 0.15  # Width of the bars
r1 = np.arange(len(alpha_values))

for beta_index, beta in enumerate(beta_values):
    # Adjusting positions for each beta
    positions = [x + (beta_index * bar_width) for x in r1]
    
    bars = plt.bar(positions, averages_beta_alpha[beta_index], width=bar_width, label=f'Beta = {beta}')
    
    # Adding the average FEM simulations above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom', size=12)

# Adding labels, title and custom x-axis tick labels, etc.
plt.xlabel('Alpha')
plt.ylabel('Average # FEM simulations')
plt.title('Average FEM simulations vs Alpha for different Beta values')
plt.xticks([r + bar_width for r in range(len(alpha_values))], [str(alpha) for alpha in alpha_values])

plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.legend()
plt.show()




# Calculate the average of each sublist
averages = [np.mean(sublist) for sublist in FEM_counter_list]

# Plotting
plt.figure()  # Set the figure size as desired
bar_width=0.1
plt.bar(alpha_values, averages, width=bar_width,color='skyblue')  # Use a bar chart to display averages

# Adding labels and title
plt.xlabel('Alpha value')
plt.ylabel('# FEM simulations')
plt.title('FEM simulations vs Alpha')
plt.xticks(alpha_values, [str(alpha) for alpha in alpha_values])

# Optionally, you can add the actual average above each bar
for i, avg in enumerate(averages):
    plt.text(alpha_values[i], avg, f'{avg:.0f}', ha='center', va='bottom')

# Show the plot
# plt.tight_layout()  # Adjust the padding between and around subplots

plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.grid(axis='x', linestyle='-', alpha=0.7)
plt.show()



" GRAPH 2: plotting the mean/std results value of all the chosen and min nodes (wrt to alpha)"

# Assuming beta_values, min_nodes, and attr_to_minimize are defined
min_nodes_attribute_values = [[node[attr_to_minimize] for node in sublist] for sublist in min_nodes_list]

# Calculate mean and standard deviation
min_stats = [stats.norm.fit(sublist) for sublist in min_nodes_attribute_values]  # This gives a list of tuples (mean, std)

# Unpack the mean and standard deviation
means = [stat[0] for stat in min_stats]
std_devs = [stat[1] for stat in min_stats]

plt.errorbar(alpha_values, means, yerr=std_devs, fmt='x', label='Mean with Std Dev', color='blue', ecolor='gray', elinewidth=3, capsize=5)

# if stat_type == 'Avg':
    # Determine y-values for percentile calculations (using ylims)
y_values = np.linspace(plt.ylim()[0], plt.ylim()[1], 6)

# Adding horizontal lines and text for each y-value
for y_val in y_values[0:]:
    percentile = 100 - stats.percentileofscore(results_exhaustive_terminal, y_val)
    plt.hlines(y_val, 0, max(alpha_values), colors='gray', linestyles='dashed')
    plt.text(max(alpha_values), y_val, f'{percentile:.2f}%', va='bottom', ha='right', color='gray')

# Add a horizontal line for the global minimum attribute value
plt.axhline(y=min_value_exhaust, color='red', linestyle='dashed', linewidth=2)
plt.text(max(alpha_values), min_value_exhaust, f'Global Minimum: {min_value_exhaust:.4f} (100%)', va='top', ha='right', color='red')

plt.xlabel('Alpha')
plt.ylabel('Displacement')
plt.title('Displacement vs Alpha')
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.grid(axis='x', linestyle='-', alpha=0.7)
# plt.legend(loc='lower left')
plt.show()



" GRAPH 3: Plotting the elapsed times vs alpha "
# time_means = [np.mean(sublist) for sublist in elapsed_times]
# time_stds = [np.std(sublist) for sublist in elapsed_times]

# # Plotting mean elapsed times with standard deviation error bars
# plt.errorbar(alpha_values, time_means, yerr=time_stds, fmt='o', label='Elapsed Time with Std Dev', color='blue', ecolor='gray', elinewidth=3, capsize=5)

# plt.xlabel('Alpha')
# plt.ylabel('Elapsed Time [s]')
# plt.title('Elapsed Time vs Alpha')
# plt.grid(axis='y', linestyle='-', alpha=0.7)
# plt.grid(axis='x', linestyle='-', alpha=0.7)
# plt.xticks(alpha_values, [str(alpha) for alpha in alpha_values])

# # plt.legend()
# plt.show()


" GRAPH 4: Plotting the simulation convergence episodes vs alpha "
# cutoff = 10000  # Define the cutoff index for x values you want to plot

# # Iterate over each set of sublists
# for index, sublists in enumerate(convergence_points):
#     plt.figure()
#     means, stds = calculate_means_stds(sublists)
#     x_values = list(range(len(sublists[0])))
    
#     # Limit to the cutoff index
#     if len(x_values) > cutoff:
#         x_values = x_values[:cutoff]
#         means = means[:cutoff]
#         stds = stds[:cutoff]
        
#     plt.plot(x_values, means, label=f'alpha: {alpha_values[index]:.2f}')
#     plt.fill_between(x_values, np.subtract(means, stds), np.add(means, stds), alpha=0.2)

#     # Add a horizontal line for the global minimum attribute value
#     plt.axhline(y=min_value_exhaust, color='red', linestyle='dashed', linewidth=2)
#     plt.text(max(x_values), min_value_exhaust, f'Global Minimum: {min_value_exhaust:.4f} (100%)', va='top', ha='right', color='red')
    
        
#     plt.xlabel('Episode')
#     plt.ylabel('Displacement')
#     plt.title('Min Displacement Result vs Episodes')
#     plt.legend(loc='upper right')
#     plt.grid(axis='y', linestyle='-', alpha=0.7)
#     plt.grid(axis='x', linestyle='-', alpha=0.7)
#     plt.ylim([0.035, 0.08])
#     # plt.xlim([15000, 20000])
    
#     plt.show()


" GRAPH 4: Plotting the simulation convergence episodes vs alpha "

# Placeholder function for calculating means and standard deviations
def calculate_means_stds(sublists):
    means = np.mean(sublists, axis=0)
    stds = np.std(sublists, axis=0)
    return means, stds

cutoff = 10000  # Reduced cutoff for demonstration

# Iterating over each alpha value
for alpha_index in range(len(alpha_values)):
    plt.figure(figsize=(10, 6))
    
    # Iterate over beta values for the same alpha index
    for beta_index, beta_sublists in enumerate(convergence_points):
        means, stds = calculate_means_stds([sublist[alpha_index] for sublist in beta_sublists])
        x_values = list(range(len(means)))
        
        # Limit to the cutoff index
        if len(x_values) > cutoff:
            x_values = x_values[:cutoff]
            means = means[:cutoff]
            stds = stds[:cutoff]
        
        plt.plot(x_values, means, label=f'Beta: {beta_values[beta_index]:.1f}')
        plt.fill_between(x_values, np.subtract(means, stds), np.add(means, stds), alpha=0.2)
    
    # Add a horizontal line for the global minimum attribute value
    plt.axhline(y=min_value_exhaust, color='red', linestyle='dashed', linewidth=2)
    plt.text(max(x_values), min_value_exhaust, f'Global Min: {min_value_exhaust:.4f}', va='top', ha='right', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Displacement')
    plt.title(f'Min Displacement Result vs Episodes (Alpha: {alpha_values[alpha_index]:.1f})')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='-', alpha=0.7)
    plt.ylim([0.55, 1.05])  # Placeholder limits
    
    plt.show()


# for index, sublists in enumerate(convergence_points):
#     plt.figure()  # Optional: Adjust figure size
#     # Calculate means and standard deviations for plotting
#     means, stds = calculate_means_stds(sublists)
#     x_values = list(range(len(sublists[0])))

#     # Limit to the cutoff index
#     if len(x_values) > cutoff:
#         x_values = x_values[:cutoff]
#         means = means[:cutoff]
#         stds = stds[:cutoff]
    
    
#     # Plot every simulation run in grey
#     for sublist in sublists:
#         if len(sublist) > cutoff:
#             sublist = sublist[:cutoff]

#         plt.plot(x_values, sublist, color=colors[index],alpha=0.3, linewidth=1)

#     # Overlay the mean in blue
#     plt.plot(x_values, means, color=colors[index], label=f'alpha value {alpha_values[index]:.2f}', linewidth=2)

#     # Add a horizontal line for the global minimum attribute value
#     plt.axhline(y=min_value_exhaust, color='red', linestyle='dashed', linewidth=2)
#     plt.text(max(x_values), min_value_exhaust, f'Global Minimum: {min_value_exhaust:.4f} (100%)', va='top', ha='right', color='red')
    
    
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.title('Min Result vs Episodes vs. alpha')
#     plt.legend(loc='upper right')
    
#     plt.show()


" GRAPH 4: Exhaustive Search histogram w/ 95 percentile result"


# def histogram_plot(array, color, results_exhaustive):
#     # Plot the histogram
#     count, bins, ignored = plt.hist(array, bins=20, density=True, color=color, alpha=0.5, label='Min Node')
#     mu_min, sigma_min = stats.norm.fit(array)

#     # Find the maximum value in the results_exhaustive which corresponds to 100% percentile
#     min_value = min(results_exhaustive)

#     # Adjust x_min and x_max based on the histogram and the 100% value
#     x_min, x_max = plt.xlim()

#     # Generate x_range only up to the maximum value of interest
#     x_range = np.linspace(min_value, x_max, 100)
#     p_min = stats.norm.pdf(x_range, mu_min, sigma_min)
#     plt.plot(x_range, p_min, color, linewidth=2)

#     # Drawing vertical lines at each bin edge and annotating them
#     for bin_edge in bins:
#         percentile = stats.percentileofscore(results_exhaustive, bin_edge)
#         plt.axvline(bin_edge, color='gray', linestyle='dotted')
#         plt.text(bin_edge, plt.ylim()[1] * 0.95, f'{(100-percentile):.2f}%', rotation=90, va='top', color='gray')

#     plt.xlim(x_min, x_max)  # Adjust the x-axis limit to ensure it aligns with the truncated normal distribution


# filtered_values = [value for value in results_exhaustive_terminal if value < 2*env.initialDisplacement]

# histogram_plot(filtered_values, 'k', results_exhaustive_terminal)
# plt.title("MCTS Exhaustive search histogram")
# plt.xlabel('Displacement')
# plt.ylabel('Probability')
# # plt.xlim([0.04, 0.06])
# plt.show()










