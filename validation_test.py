" ------------   IMPORT STATEMENTS  -------------- "
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import scipy.stats as stats
import json
# from functions_MCTS_print       import histogram_plot
from functions   import percentage_optimal_states #, calculate_means_stds
    from env_truss                  import env_truss


import os
import logging
from typing import Dict, Any, List

# ------------   SETUP LOGGING  -------------- 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_results(
    example: str,
    attr_to_minimize: str,
    alpha: float,
    beta: float,
    total_sims: int,
    num_eps: int,
    output_dir: str = "validation"
) -> Dict[str, Any]:
    """
    Load simulation results from a single JSON file.

    Args:
        example (str): The example name (e.g., 'bridge').
        attr_to_minimize (str): The attribute being minimized.
        alpha (float): Alpha parameter.
        beta (float): Beta parameter.
        total_sims (int): Number of simulations.
        num_eps (int): Number of episodes per simulation.
        output_dir (str, optional): Base directory where results are saved. Defaults to "validation".

    Returns:
        Dict[str, Any]: Loaded simulation results.
    """
    # Define file name
    json_file_name = f"{example}_results_{attr_to_minimize}_{alpha}_{beta}_{total_sims}_{num_eps}.json"

    # Define full path
    json_save_path = os.path.join(output_dir, example, json_file_name)

    try:
        with open(json_save_path, 'r') as json_file:
            data = json.load(json_file)
        logger.info(f"Results loaded from {json_save_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {json_save_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        return {}



def validation_test():
    """
    Perform validation tests by loading simulation results and analyzing them.
    """
    
    example                 = "Ororbia_1"
    env                     = env_truss(example)
    attr_to_minimize        = 'maxDisplacement'
    example = 'Ororbia_1'
    attr_to_minimize = 'max_displacement'
    alpha = 0.3
    beta = 0.0
    total_sims = 10
    num_eps = 1000
    
    def calculate_means_stds(sublists):
        means = np.mean(sublists, axis=0)
        stds = np.std(sublists, axis=0)
        return means, stds
    " ---------  MCTS EXHAUSTIVE ----------- "
    
    results_exhaustive_terminal = np.load(f"validation/{example}/{example}_exhaustive_terminal.npy")
    min_value = min(results_exhaustive_terminal)
    
    exhaustive_terminal_min_nodes_path          = f"validation/{example}/{example}_exhaustive_terminal_min_nodes"
    
    with open(f'{exhaustive_terminal_min_nodes_path}.json', 'r') as json_file:
        min_node_exhaustive = json.load(json_file)
    
    # Find the min value of maxDisplacement
    min_value_exhaust = min_node_exhaustive[-1]['attribute_value']
    min_state_exhaust = min_node_exhaustive[-1]['state']
    
    # min_value_exhaust = 7.147
    
    
    # print(f"Min maxDisplacement value: {min_value_exhaust}")
    # print(f"State associated with min maxDisplacement: {min_state_exhaust}")
    # env.render0(min_state_exhaust, "Global Minimum configuration")
    # env.render0_black(min_state_exhaust, r'$s_{3}$')
    
    
    

    # Load results
    results = load_results(
        example=example,
        attr_to_minimize=attr_to_minimize,
        alpha=alpha,
        beta=beta,
        total_sims=total_sims,
        num_eps=num_eps,
        output_dir="validation"
    )

    if not results:
        logger.error("No results to process.")
        return

    # Extract data
    min_nodes = results.get("results", {}).get("min_nodes", [])
    elapsed_times = results.get("results", {}).get("elapsed_times", [])
    convergence_points = results.get("results", {}).get("min_result_episodes", [])
    FEM_counters = results.get("results", {}).get("FEM_counters", [])

    # Example: Print summary
    print(f"Total Simulations: {len(min_nodes)}")
    if elapsed_times:
        avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
        print(f"Average Elapsed Time: {avg_elapsed_time:.4f} seconds")
    print(f"FEM Simulations per Run: {FEM_counters}")

    # Example: Accessing min_nodes data
    for idx, node in enumerate(min_nodes, start=1):
        print(f"Simulation {idx}:")
        print(f"  Max Displacement: {node['max_displacement']}")
        print(f"  State: {node['state']}")
        print()

    # Further analysis can be done here
    # For example, plotting, statistical analysis, etc.





if __name__ == "__main__":
    validation_test()



" -------------  G R A P H S ------------- "
for min_node in min_node_exhaustive:
    env.render0_black(min_node['state'])
    print(min_node['attribute_value'])

# env.render0_black(env.initial_state)

# for min_node in min_nodes_list[0]:
#     env.render0_black(min_node['state'])
#     print(min_node['maxDisplacement'])





# Set the font properties globally
matplotlib.rc('font', family='Arial', size=24)
colors = ['blue', 'orange', 'green', 'red','purple', 'brown']

def truncate(value, decimals=8):
    return float(f"{value:.{decimals}f}")

" GRAPH 1: Plotting the percentile of the min_nodes wrt to alpha"



# plt.figure(figsize=(10, 6))

# percentile_scores = []
# error_bars = []

# for i in range(len(min_nodes_list)):
    
#     # Get attribute values for min_nodes
#     attribute_values = [node['maxDisplacement'] for node in min_nodes_list[i]]
    
#     # Calculate the mean and standard deviation of attribute values
#     mean_attr_value = np.mean(attribute_values)
#     std_attr_value = np.std(attribute_values)
    
#     # Truncate the mean and std values
#     mean_attr_value = truncate(mean_attr_value, 7)
#     std_attr_value = truncate(std_attr_value, 7)
    
#     # Calculate the percentile score for the mean attribute value
#     mean_percentile_score = 100 - stats.percentileofscore(results_exhaustive_terminal, mean_attr_value)
#     percentile_scores.append(mean_percentile_score)
    
#     # Calculate the upper and lower bounds for the attribute value
#     upper_value = mean_attr_value + std_attr_value
#     lower_value = mean_attr_value - std_attr_value
    
#     # Calculate percentile scores for the upper and lower bounds
#     upper_percentile = 100 - stats.percentileofscore(results_exhaustive_terminal, upper_value)
#     lower_percentile = 100 - stats.percentileofscore(results_exhaustive_terminal, lower_value)
    
#     # Calculate the error as half the difference between upper and lower percentile scores
#     error_bar = (lower_percentile - upper_percentile) / 2
#     error_bars.append(error_bar)
    
# # Create the plot with error bars
# plt.errorbar(alpha_values, percentile_scores, yerr=error_bars, fmt='o', label='Percentile Score with Std Dev', color='blue', ecolor='gray', elinewidth=5, capsize=8)

# # Add labels and title
# plt.xlabel('$\\alpha$')
# plt.ylabel('Percentile Score [%]')
# plt.xticks(alpha_values, [str(alpha) for alpha in alpha_values])

# # Ensure 100 is included on the y-axis
# plt.ylim([98, 100])
# plt.yticks(np.arange(98, 100.5, 0.5))  # Creates ticks at intervals of 0.5 from 98 to 100

# # Add grid and legend
# plt.grid(axis='y', linestyle='-', alpha=0.7)
# plt.grid(axis='x', linestyle='-', alpha=0.7)

# plt.show()



" GRAPH 1: Plotting the percentage of found optimal states by the algorithm"
# percental_optimal_min_nodes     = percentage_optimal_states(min_nodes_list, [min_state_exhaust], total_simulations)
# # Convert the line plot to a column graph
# plt.figure()
# bar_width = 0.1  # Width of the bars
# positions = np.arange(len(alpha_values))  # Position of bars on x-axis

# plt.bar(positions, percental_optimal_min_nodes, width=bar_width)

# # Adding labels, title, and custom x-axis tick labels
# plt.xlabel('Alpha')
# plt.ylabel('Percentage of Optimal States found %')
# plt.title('Percentage of optimal states found vs Alpha')
# plt.xticks(positions, [str(alpha) for alpha in alpha_values])

# # Optionally, add the percentage above each bar
# for i, percentage in enumerate(percental_optimal_min_nodes):
#     plt.text(i, percentage + 1, f'{percentage}%', ha='center', va='bottom')


# y_max = max(percental_optimal_min_nodes) + 5  # Extend 10 units below the minimum score, or to 0

# plt.ylim([0, y_max])
# plt.grid(axis='y', linestyle='-', alpha=0.7)
# # plt.legend()
# plt.show()




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

# # Define the optimal results for each example
# optimal_results = {
#     'Ororbia_1': 0.0895,
#     'Ororbia_2': 0.1895,
#     'Ororbia_3': 0.03606260275840524,
#     'Ororbia_4': 0.5915955878577671,
#     'Ororbia_5': 0.039030144110207686,
#     'Ororbia_7': 0.042
# }

# # Function to calculate the percentage error
# def calculate_percentage_error(values, optimal_result):
#     return [optimal_result / value * 100 for value in values]

# # Iterate over each set of sublists and plot the percentage error
# plt.figure(figsize=(10, 6))
# for index, sublists in enumerate(convergence_points):
#     means, stds = calculate_means_stds(sublists)
#     x_values = list(range(len(sublists[0])))
    
#     # Retrieve the optimal result for the current example
#     optimal_result = optimal_results[example]
    
#     # Calculate the percentage error
#     means_percentage_error = calculate_percentage_error(means, optimal_result)
#     stds_percentage_error = [std / optimal_result * 100 for std in stds]

#     # Plot the percentage error and fill the error band
#     plt.plot(x_values, means_percentage_error, label=f'$\\alpha$: {alpha_values[index]:.2f}', linewidth=3)
#     plt.fill_between(x_values, 
#                      np.subtract(means_percentage_error, stds_percentage_error), 
#                      np.add(means_percentage_error, stds_percentage_error), 
#                      alpha=0.2)

# plt.xlabel('Episode')
# plt.ylabel('Displacement Error (%)')
# plt.legend(loc='lower right', fontsize='32')
# plt.ylim([50, 100])  # Adjust the y-axis limits as needed
# # ex.1
# # plt.xlim([0, 150])
# # # ex.2
# # plt.xlim([0, 400])
# # # ex.3

# plt.show()

# print(f'the mean final displacement percentage  = {means_percentage_error[-1]}')


" GRAPH 4: Plotting the simulation convergence episodes vs alpha "

# Iterate over each set of sublists
plt.figure(figsize=(10, 6))
for index, sublists in enumerate(convergence_points):
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

# ex.1
# plt.yticks(np.arange(0.08, 0.19, 0.02))
# plt.xlim([0, 150])
# plt.ylim([0.08, 0.18])
# # ex.2
# plt.yticks(np.arange(0.17, 0.31, 0.02))
# plt.ylim([0.17, 0.3])
# plt.xlim([0, 400])
# # ex.3
# plt.yticks(np.arange(0.03, 0.09, 0.01))
# plt.ylim([0.03, 0.08])
# plt.xlim([0, 1000])
# # ex.4
# plt.yticks(np.arange(0.6, 1.16, 0.1))
# plt.ylim([0.55, 1.18])
# # ex.5
# plt.yticks(np.arange(0.035, 0.08, 0.01))
# plt.ylim([0.035, 0.08])
# # ex.6
# plt.yticks(np.arange(0.04, 0.1, 0.01))
# plt.ylim([0.035, 0.1])


# bridge
plt.yticks(np.arange(7, 9.2, 0.5))
plt.ylim([6.9, 9.2])
plt.xlim([0, 1000])

plt.show()


" GRAPH 4: Plotting the simulation convergence episodes vs alpha "
# plt.figure(figsize=(10, 6))

# # Assuming beta_values, min_nodes, and attr_to_minimize are defined
# min_nodes_attribute_values = [[node[attr_to_minimize] for node in sublist] for sublist in min_nodes_list]

# # Calculate mean and standard deviation
# min_stats = [stats.norm.fit(sublist) for sublist in min_nodes_attribute_values]  # This gives a list of tuples (mean, std)

# # Unpack the mean and standard deviation
# means = [stat[0] for stat in min_stats]
# std_devs = [stat[1] for stat in min_stats]

# plt.errorbar(alpha_values, means, yerr=std_devs, fmt='x', label='Mean with Std Dev', color='blue', ecolor='gray', elinewidth=5, capsize=8)

# y_values = np.linspace(plt.ylim()[0], plt.ylim()[1], 6)

# # Adding horizontal lines and text for each y-value
# for y_val in y_values[1:5]:
#     percentile = 100 - stats.percentileofscore(results_exhaustive_terminal, y_val)
#     plt.hlines(y_val, 0, max(alpha_values)+0.05, colors='gray', linestyles='dashed')
#     plt.text(max(alpha_values)+0.05, y_val, f'{percentile:.2f}%', va='bottom', ha='right', color='gray')

# # Add a horizontal line for the global minimum attribute value
# plt.axhline(y=min_value_exhaust, color='red', linestyle='dashed', linewidth=3)

# plt.xlabel('$\\alpha$')
# plt.ylabel('Displacement')
# # plt.title('Displacement vs Alpha')
# # # ex.4
# plt.yticks(np.arange(0.6, 0.96, 0.05))
# plt.ylim([0.58, 0.98])
# plt.xlim([0.05, 0.55])
# # # ex.6
# # plt.yticks(np.arange(0.04, 0.11, 0.01))
# # plt.ylim([0.035, 0.1])

# plt.show()


" GRAPH PERCENTILE SCORES FOR DIFFERENT ALPHAS" 
# Assuming convergence_points, alpha_values, and results_exhaustive_terminal are defined
# Calculate and store mean and std of percentile scores for each alpha

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats

# def truncate(value, decimals=8):
#     return float(f"{value:.{decimals}f}")

# def truncate_sublists(sublists, decimals=8):
#     return [[truncate(item, decimals) for item in sublist] for sublist in sublists]

# # Initialize percentile_statistics
# percentile_statistics = []

# # Iterate over all convergence_points
# for index, convergence_point in enumerate(convergence_points):
    
#     # Truncate the values for the current convergence_point
#     truncated_convergence_points = truncate_sublists(convergence_point, 7)
    
#     # Convert the list of lists into a numpy array for easier manipulation
#     truncated_convergence_points_array = np.array(truncated_convergence_points)
    
#     # Calculate mean and std deviation across the 10 lists at each episode (across axis 0)
#     episode_means = np.mean(truncated_convergence_points_array, axis=0)
#     episode_stds = np.std(truncated_convergence_points_array, axis=0)
    
#     # Convert the mean values to percentile scores
#     episode_percentiles = [100 - stats.percentileofscore(results_exhaustive_terminal, mean_value) for mean_value in episode_means]
    
#     # Calculate the standard deviation of percentile scores using the original std deviation
#     std_percentiles = []
#     for i in range(len(episode_means)):
#         upper_value = episode_means[i] + episode_stds[i]
#         lower_value = episode_means[i] - episode_stds[i]
#         upper_percentile = 100 - stats.percentileofscore(results_exhaustive_terminal, upper_value)
#         lower_percentile = 100 - stats.percentileofscore(results_exhaustive_terminal, lower_value)
#         std_percentiles.append((upper_percentile - lower_percentile) / 2)
    
#     # Append the result to percentile_statistics
#     percentile_statistics.append((episode_percentiles, std_percentiles))

# # Plotting
# matplotlib.rc('font', family='Arial', size='28')  # Set font globally

# plt.figure(figsize=(10, 6))
# for index, (mean_percentiles, std_percentiles) in enumerate(percentile_statistics):
#     x_values = list(range(len(mean_percentiles)))
    
#     # Plot the mean percentile scores and their variability
#     plt.plot(x_values, mean_percentiles, label=f'$\\alpha$: {alpha_values[index]}', linewidth=3)
#     plt.fill_between(x_values, np.array(mean_percentiles) - np.array(std_percentiles), 
#                      np.array(mean_percentiles) + np.array(std_percentiles), alpha=0.2)

# plt.xlabel('Episode')
# plt.ylabel('Percentile Score [%]')
# plt.ylim([98, 100])  # Adjusted to the typical range of percentile scores
# # plt.xlim([0, 10000])
# plt.legend(loc='lower right', fontsize='20')
# plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats

# def truncate(value, decimals=8):
#     return float(f"{value:.{decimals}f}")

# def truncate_sublists(sublists, decimals=8):
#     return [[truncate(item, decimals) for item in sublist] for sublist in sublists]

# # Initialize percentile_statistics
# percentile_statistics = []

# # Iterate over all convergence_points
# for index, convergence_point in enumerate(convergence_points):
    
#     # Truncate the values for the current convergence_point
#     truncated_convergence_points = truncate_sublists(convergence_point, 6)
    
#     # Calculate episode percentiles
#     episode_percentiles = []
#     for episode_data in zip(*truncated_convergence_points):  # Transpose to get episode-wise data
#         episode_scores = [100 - stats.percentileofscore(results_exhaustive_terminal, value) for value in episode_data]
#         episode_percentiles.append(episode_scores)
    
#     # Calculate mean and std deviation of percentiles
#     mean_percentiles = np.mean(episode_percentiles, axis=1)
#     std_percentiles = np.std(episode_percentiles, axis=1)
    
#     # Append the result to percentile_statistics
#     percentile_statistics.append((mean_percentiles, std_percentiles))

# # Plotting
# matplotlib.rc('font', family='Arial', size='28')  # Set font globally

# plt.figure(figsize=(10, 6))
# for index, (mean_percentiles, std_percentiles) in enumerate(percentile_statistics):
#     x_values = list(range(len(mean_percentiles)))
    
#     # Plot the mean percentile scores and their variability
#     plt.plot(x_values, mean_percentiles, label=f'$\\alpha$: {alpha_values[index]}', linewidth=3)
#     plt.fill_between(x_values, mean_percentiles - std_percentiles, mean_percentiles + std_percentiles, alpha=0.2)

# plt.xlabel('Episode')
# plt.ylabel('Percentile Score [%]')
# plt.ylim([98, 100])  # Adjusted to the typical range of percentile scores
# plt.xlim([0, 1000])
# plt.legend(loc='lower right', fontsize='20')
# plt.show()






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
#     # for bin_edge in bins:
#     #     percentile = stats.percentileofscore(results_exhaustive, bin_edge)
#     #     plt.axvline(bin_edge, color='gray', linestyle='dotted')
#     #     plt.text(bin_edge, plt.ylim()[1] * 0.95, f'{(100-percentile):.2f}%', rotation=90, va='top', color='gray')

#     plt.xlim(x_min, x_max)  # Adjust the x-axis limit to ensure it aligns with the truncated normal distribution

# plt.figure(figsize=(10, 6))
# # plt.grid(axis='y', linestyle='-', alpha=0.7)
# # plt.grid(axis='x', linestyle='-', alpha=0.7)

# filtered_values = [value for value in results_exhaustive_terminal if value < 4*env.initialDisplacement]

# histogram_plot(filtered_values, 'k', results_exhaustive_terminal)
# # plt.title("MCTS Exhaustive search histogram")
# plt.xlabel('Displacement')
# plt.ylabel('Probability')

# # plt.xlim([0.04, 0.06])
# plt.show()










