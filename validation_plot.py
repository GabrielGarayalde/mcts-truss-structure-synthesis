import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
import numpy as np
from functions   import percentage_optimal_states #, calculate_means_stds


# Set the font properties globally
matplotlib.rc('font', family='Arial', size=24)
colors = ['blue', 'orange', 'green', 'red','purple', 'brown']

def truncate(value, decimals=8):
    return float(f"{value:.{decimals}f}")


" GRAPH 1: Plotting the percentile of the min_nodes wrt to alpha"
def plot_percentile_score(results_exhaustive_terminal, min_nodes_list, alpha_values):

    plt.figure(figsize=(10, 6))
    
    percentile_scores = []
    error_bars = []
    
    for i in range(len(min_nodes_list)):
        
        # Get attribute values for min_nodes
        attribute_values = [node['maxDisplacement'] for node in min_nodes_list[i]]
        
        # Calculate the mean and standard deviation of attribute values
        mean_attr_value = np.mean(attribute_values)
        std_attr_value = np.std(attribute_values)
        
        # Truncate the mean and std values
        mean_attr_value = truncate(mean_attr_value, 7)
        std_attr_value = truncate(std_attr_value, 7)
        
        # Calculate the percentile score for the mean attribute value
        mean_percentile_score = 100 - stats.percentileofscore(results_exhaustive_terminal, mean_attr_value)
        percentile_scores.append(mean_percentile_score)
        
        # Calculate the upper and lower bounds for the attribute value
        upper_value = mean_attr_value + std_attr_value
        lower_value = mean_attr_value - std_attr_value
        
        # Calculate percentile scores for the upper and lower bounds
        upper_percentile = 100 - stats.percentileofscore(results_exhaustive_terminal, upper_value)
        lower_percentile = 100 - stats.percentileofscore(results_exhaustive_terminal, lower_value)
        
        # Calculate the error as half the difference between upper and lower percentile scores
        error_bar = (lower_percentile - upper_percentile) / 2
        error_bars.append(error_bar)
        
    # Create the plot with error bars
    plt.errorbar(alpha_values, percentile_scores, yerr=error_bars, fmt='o', label='Percentile Score with Std Dev', color='blue', ecolor='gray', elinewidth=5, capsize=8)
    
    # Add labels and title
    plt.xlabel('$\\alpha$')
    plt.ylabel('Percentile Score [%]')
    plt.xticks(alpha_values, [str(alpha) for alpha in alpha_values])
    
    # Ensure 100 is included on the y-axis
    plt.ylim([98, 100])
    plt.yticks(np.arange(98, 100.5, 0.5))  # Creates ticks at intervals of 0.5 from 98 to 100
    
    # Add grid and legend
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='-', alpha=0.7)
    
    plt.show()



" GRAPH 1: Plotting the percentage of found optimal states by the algorithm"

def percentage_optimal_nodes():
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

def plot_fem_evaluations():
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
def plot_elapsed_times():
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

def plot_percentage_result():
    # Define the optimal results for each example
    optimal_results = {
        'Ororbia_1': 0.0895,
        'Ororbia_2': 0.1895,
        'Ororbia_3': 0.03606260275840524,
        'Ororbia_4': 0.5915955878577671,
        'Ororbia_5': 0.039030144110207686,
        'Ororbia_7': 0.042
    }
    
    # Function to calculate the percentage error
    def calculate_percentage_error(values, optimal_result):
        return [optimal_result / value * 100 for value in values]
    
    # Iterate over each set of sublists and plot the percentage error
    plt.figure(figsize=(10, 6))
    for index, sublists in enumerate(convergence_points):
        means, stds = calculate_means_stds(sublists)
        x_values = list(range(len(sublists[0])))
        
        # Retrieve the optimal result for the current example
        optimal_result = optimal_results[example]
        
        # Calculate the percentage error
        means_percentage_error = calculate_percentage_error(means, optimal_result)
        stds_percentage_error = [std / optimal_result * 100 for std in stds]
    
        # Plot the percentage error and fill the error band
        plt.plot(x_values, means_percentage_error, label=f'$\\alpha$: {alpha_values[index]:.2f}', linewidth=3)
        plt.fill_between(x_values, 
                          np.subtract(means_percentage_error, stds_percentage_error), 
                          np.add(means_percentage_error, stds_percentage_error), 
                          alpha=0.2)
    
    plt.xlabel('Episode')
    plt.ylabel('Displacement Error (%)')
    plt.legend(loc='lower right', fontsize='32')
    plt.ylim([50, 100])  # Adjust the y-axis limits as needed
    # ex.1
    # plt.xlim([0, 150])
    # # ex.2
    # plt.xlim([0, 400])
    # # ex.3
    
    plt.show()
    
    print(f'the mean final displacement percentage  = {means_percentage_error[-1]}')




" GRAPH 4: Plotting the simulation convergence episodes vs alpha "

def plot_result():
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





" GRAPH PERCENTILE SCORES FOR DIFFERENT ALPHAS" 

# Assuming convergence_points, alpha_values, and results_exhaustive_terminal are defined
# Calculate and store mean and std of percentile scores for each alpha

def plot_percentile_results():
    def truncate(value, decimals=8):
        return float(f"{value:.{decimals}f}")
    
    def truncate_sublists(sublists, decimals=8):
        return [[truncate(item, decimals) for item in sublist] for sublist in sublists]

    # Initialize percentile_statistics
    percentile_statistics = []
    
    # Iterate over all convergence_points
    for index, convergence_point in enumerate(convergence_points):
        
        # Truncate the values for the current convergence_point
        truncated_convergence_points = truncate_sublists(convergence_point, 7)
        
        # Convert the list of lists into a numpy array for easier manipulation
        truncated_convergence_points_array = np.array(truncated_convergence_points)
        
        # Calculate mean and std deviation across the 10 lists at each episode (across axis 0)
        episode_means = np.mean(truncated_convergence_points_array, axis=0)
        episode_stds = np.std(truncated_convergence_points_array, axis=0)
        
        # Convert the mean values to percentile scores
        episode_percentiles = [100 - stats.percentileofscore(results_exhaustive_terminal, mean_value) for mean_value in episode_means]
        
        # Calculate the standard deviation of percentile scores using the original std deviation
        std_percentiles = []
        for i in range(len(episode_means)):
            upper_value = episode_means[i] + episode_stds[i]
            lower_value = episode_means[i] - episode_stds[i]
            upper_percentile = 100 - stats.percentileofscore(results_exhaustive_terminal, upper_value)
            lower_percentile = 100 - stats.percentileofscore(results_exhaustive_terminal, lower_value)
            std_percentiles.append((upper_percentile - lower_percentile) / 2)
        
        # Append the result to percentile_statistics
        percentile_statistics.append((episode_percentiles, std_percentiles))
    
    # Plotting
    matplotlib.rc('font', family='Arial', size='28')  # Set font globally
    
    plt.figure(figsize=(10, 6))
    for index, (mean_percentiles, std_percentiles) in enumerate(percentile_statistics):
        x_values = list(range(len(mean_percentiles)))
        
        # Plot the mean percentile scores and their variability
        plt.plot(x_values, mean_percentiles, label=f'$\\alpha$: {alpha_values[index]}', linewidth=3)
        plt.fill_between(x_values, np.array(mean_percentiles) - np.array(std_percentiles), 
                          np.array(mean_percentiles) + np.array(std_percentiles), alpha=0.2)
    
    plt.xlabel('Episode')
    plt.ylabel('Percentile Score [%]')
    plt.ylim([98, 100])  # Adjusted to the typical range of percentile scores
    # plt.xlim([0, 10000])
    plt.legend(loc='lower right', fontsize='20')
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_percentile_results_2():

    def truncate(value, decimals=8):
        return float(f"{value:.{decimals}f}")
    
    def truncate_sublists(sublists, decimals=8):
        return [[truncate(item, decimals) for item in sublist] for sublist in sublists]
    
    # Initialize percentile_statistics
    percentile_statistics = []
    
    # Iterate over all convergence_points
    for index, convergence_point in enumerate(convergence_points):
        
        # Truncate the values for the current convergence_point
        truncated_convergence_points = truncate_sublists(convergence_point, 6)
        
        # Calculate episode percentiles
        episode_percentiles = []
        for episode_data in zip(*truncated_convergence_points):  # Transpose to get episode-wise data
            episode_scores = [100 - stats.percentileofscore(results_exhaustive_terminal, value) for value in episode_data]
            episode_percentiles.append(episode_scores)
        
        # Calculate mean and std deviation of percentiles
        mean_percentiles = np.mean(episode_percentiles, axis=1)
        std_percentiles = np.std(episode_percentiles, axis=1)
        
        # Append the result to percentile_statistics
        percentile_statistics.append((mean_percentiles, std_percentiles))
    
    # Plotting
    matplotlib.rc('font', family='Arial', size='28')  # Set font globally
    
    plt.figure(figsize=(10, 6))
    for index, (mean_percentiles, std_percentiles) in enumerate(percentile_statistics):
        x_values = list(range(len(mean_percentiles)))
        
        # Plot the mean percentile scores and their variability
        plt.plot(x_values, mean_percentiles, label=f'$\\alpha$: {alpha_values[index]}', linewidth=3)
        plt.fill_between(x_values, mean_percentiles - std_percentiles, mean_percentiles + std_percentiles, alpha=0.2)
    
    plt.xlabel('Episode')
    plt.ylabel('Percentile Score [%]')
    plt.ylim([98, 100])  # Adjusted to the typical range of percentile scores
    plt.xlim([0, 1000])
    plt.legend(loc='lower right', fontsize='20')
    plt.show()




