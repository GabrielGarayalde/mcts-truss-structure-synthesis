# validation_plots.py

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any

# Set the font properties globally
matplotlib.rc('font', family='Arial', size=24)


def truncate(value: float, decimals: int = 8) -> float:
    """
    Truncate a float to a specified number of decimal places.

    Args:
        value (float): The value to truncate.
        decimals (int): Number of decimal places.

    Returns:
        float: The truncated float.
    """
    return float(f"{value:.{decimals}f}")


def calculate_means_stds(sublists: List[List[float]]) -> (np.ndarray, np.ndarray):
    """
    Calculate means and standard deviations across sublists.

    Args:
        sublists (List[List[float]]): List of lists of numerical values.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of means and standard deviations.
    """
    means = np.mean(sublists, axis=0)
    stds = np.std(sublists, axis=0)
    return means, stds


def truncate_sublists(sublists: List[List[float]], decimals: int = 8) -> List[List[float]]:
    """
    Truncate each float in sublists to a specified number of decimal places.

    Args:
        sublists (List[List[float]]): List of lists of numerical values.
        decimals (int): Number of decimal places to keep.

    Returns:
        List[List[float]]: List of lists with truncated float values.
    """
    return [[truncate(item, decimals) for item in sublist] for sublist in sublists]


def plot_percentile_score(
    results_exhaustive_terminal: np.ndarray,
    min_nodes_list: List[List[Dict[str, Any]]],
    alpha_values: List[float]
):
    """
    Plot the percentile scores of min_nodes with respect to alpha values.

    Args:
        results_exhaustive_terminal (np.ndarray): The exhaustive terminal results as an array.
        min_nodes_list (List[List[Dict[str, Any]]]): List where each element corresponds to min_nodes for a particular alpha value.
        alpha_values (List[float]): List of alpha values used in the simulations.
    """
    plt.figure(figsize=(10, 6))
    percentile_scores = []
    error_bars = []

    for i in range(len(min_nodes_list)):
        # Get attribute values for min_nodes
        attribute_values = [node['max_displacement'] for node in min_nodes_list[i]]

        # Calculate the mean and standard deviation of attribute values
        mean_attr_value = truncate(np.mean(attribute_values), 7)
        std_attr_value = truncate(np.std(attribute_values), 7)

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
    plt.errorbar(
        alpha_values,
        percentile_scores,
        yerr=error_bars,
        fmt='o',
        color='blue',
        ecolor='gray',
        elinewidth=5,
        capsize=8
    )

    # Add labels and adjust axes
    plt.xlabel('$\\alpha$')
    plt.ylabel('Percentile Score [%]')
    plt.xticks(alpha_values, [str(alpha) for alpha in alpha_values])
    plt.ylim([98, 100])
    plt.yticks(np.arange(98, 100.5, 0.5))
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='-', alpha=0.7)
    plt.show()


def percentage_optimal_nodes(
    min_nodes_list: List[List[Dict[str, Any]]],
    min_state_exhaust: Any,
    total_simulations: int,
    alpha_values: List[float],
    percentage_optimal_states_function
):
    """
    Plot the percentage of optimal states found by the algorithm for different alpha values.

    Args:
        min_nodes_list (List[List[Dict[str, Any]]]): List of min_nodes for each alpha value.
        min_state_exhaust (Any): The state corresponding to the global minimum.
        total_simulations (int): Total number of simulations run for each alpha.
        alpha_values (List[float]): List of alpha values used in the simulations.
        percentage_optimal_states_function (Callable): Function to calculate the percentage of optimal states.
    """
    percental_optimal_min_nodes = percentage_optimal_states_function(
        min_nodes_list, [min_state_exhaust], total_simulations
    )

    # Convert the line plot to a column graph
    plt.figure()
    bar_width = 0.1  # Width of the bars
    positions = np.arange(len(alpha_values))  # Position of bars on x-axis

    plt.bar(positions, percental_optimal_min_nodes, width=bar_width)

    # Adding labels and custom x-axis tick labels
    plt.xlabel('Alpha')
    plt.ylabel('Percentage of Optimal States found [%]')
    plt.title('Percentage of Optimal States Found vs Alpha')
    plt.xticks(positions, [str(alpha) for alpha in alpha_values])

    # Optionally, add the percentage above each bar
    for i, percentage in enumerate(percental_optimal_min_nodes):
        plt.text(i, percentage + 1, f'{percentage}%', ha='center', va='bottom')

    y_max = max(percental_optimal_min_nodes) + 5  # Extend 5 units above the maximum percentage
    plt.ylim([0, y_max])
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.show()


def plot_fem_evaluations(
    alpha_values: List[float],
    FEM_counter_list: List[List[int]]
):
    """
    Plot the average number of FEM simulations per alpha value.

    Args:
        alpha_values (List[float]): List of alpha values used in the simulations.
        FEM_counter_list (List[List[int]]): List of FEM counters for each simulation at each alpha value.
    """
    averages = [np.mean(sublist) for sublist in FEM_counter_list]
    plt.figure(figsize=(10, 6))
    bar_width = 0.05
    plt.bar(alpha_values, averages, width=bar_width, color='skyblue')

    plt.xlabel('$\\alpha$')
    plt.ylabel('FEM Evaluations')
    plt.xticks(alpha_values, [str(alpha) for alpha in alpha_values])
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='-', alpha=0.7)
    plt.show()


def plot_elapsed_times(
    alpha_values: List[float],
    elapsed_times_list: List[List[float]]
):
    """
    Plot the mean elapsed times with standard deviation for each alpha value.

    Args:
        alpha_values (List[float]): List of alpha values used in the simulations.
        elapsed_times_list (List[List[float]]): List of elapsed times for each simulation at each alpha value.
    """
    time_means = [np.mean(sublist) for sublist in elapsed_times_list]
    time_stds = [np.std(sublist) for sublist in elapsed_times_list]

    plt.errorbar(
        alpha_values,
        time_means,
        yerr=time_stds,
        fmt='o',
        color='blue',
        ecolor='gray',
        elinewidth=3,
        capsize=5
    )

    plt.xlabel('Alpha')
    plt.ylabel('Elapsed Time [s]')
    plt.title('Elapsed Time vs Alpha')
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.grid(axis='x', linestyle='-', alpha=0.7)
    plt.xticks(alpha_values, [str(alpha) for alpha in alpha_values])
    plt.show()


def plot_percentage_result(
    convergence_points_list: List[List[List[float]]],
    alpha_values: List[float],
    example: str,
    min_value_exhaust: float
):
    """
    Plot the percentage error of displacement over episodes for different alpha values.

    Args:
        convergence_points_list (List[List[List[float]]]): List of convergence points for each alpha value.
        alpha_values (List[float]): List of alpha values used in the simulations.
        example (str): Name of the example used (e.g., 'Ororbia_1').
        min_value_exhaust (float): The global minimum value from exhaustive search.
    """
    # Define the optimal results for each example
    optimal_results = {
        'Ororbia_1': 0.0895,
        'Ororbia_2': 0.1895,
        'Ororbia_3': 0.03606260275840524,
        'Ororbia_4': 0.5915955878577671,
        'Ororbia_5': 0.039030144110207686,
        'Ororbia_7': 0.042
    }

    # Retrieve the optimal result for the current example
    optimal_result = optimal_results.get(example)
    if optimal_result is None:
        raise ValueError(f"Optimal result not defined for example '{example}'")

    # Function to calculate the percentage error
    def calculate_percentage_error(values, optimal_result):
        return [optimal_result / value * 100 for value in values]

    # Iterate over each set of sublists and plot the percentage error
    plt.figure(figsize=(10, 6))
    for index, sublists in enumerate(convergence_points_list):
        means, stds = calculate_means_stds(sublists)
        x_values = list(range(len(sublists[0])))

        # Calculate the percentage error
        means_percentage_error = calculate_percentage_error(means, optimal_result)
        stds_percentage_error = [std / optimal_result * 100 for std in stds]

        # Plot the percentage error and fill the error band
        plt.plot(
            x_values,
            means_percentage_error,
            label=f'$\\alpha$: {alpha_values[index]:.2f}',
            linewidth=3
        )
        plt.fill_between(
            x_values,
            np.subtract(means_percentage_error, stds_percentage_error),
            np.add(means_percentage_error, stds_percentage_error),
            alpha=0.2
        )

    plt.xlabel('Episode')
    plt.ylabel('Displacement Error (%)')
    plt.legend(loc='lower right', fontsize='32')
    plt.ylim([50, 100])  # Adjust the y-axis limits as needed
    plt.show()

    # Print the final mean displacement percentage error
    print(f'The mean final displacement percentage = {means_percentage_error[-1]}')


def plot_result(
    convergence_points_list: List[List[List[float]]],
    alpha_values: List[float],
    min_value_exhaust: float
):
    """
    Plot the simulation convergence over episodes for different alpha values.

    Args:
        convergence_points_list (List[List[List[float]]]): List of convergence points for each alpha value.
        alpha_values (List[float]): List of alpha values used in the simulations.
        min_value_exhaust (float): The global minimum value from exhaustive search.
    """
    plt.figure(figsize=(10, 6))
    for index, sublists in enumerate(convergence_points_list):
        means, stds = calculate_means_stds(sublists)
        x_values = list(range(len(means)))

        plt.plot(
            x_values,
            means,
            label=f'$\\alpha$: {alpha_values[index]:.2f}',
            linewidth=3
        )
        plt.fill_between(
            x_values,
            np.subtract(means, stds),
            np.add(means, stds),
            alpha=0.2
        )

    # Add a horizontal line for the global minimum attribute value
    plt.axhline(y=min_value_exhaust, color='red', linestyle='dashed', linewidth=3)

    plt.xlabel('Episode')
    plt.ylabel('Displacement')
    plt.legend(loc='upper right', fontsize='20')

    # Adjust y-axis limits as needed
    plt.yticks(np.arange(7, 9.2, 0.5))
    plt.ylim([6.9, 9.2])
    plt.xlim([0, 1000])

    plt.show()


def plot_percentile_results(
    convergence_points_list: List[List[List[float]]],
    alpha_values: List[float],
    results_exhaustive_terminal: np.ndarray
):
    """
    Plot percentile scores over episodes for different alpha values.

    Args:
        convergence_points_list (List[List[List[float]]]): List of convergence points for each alpha value.
        alpha_values (List[float]): List of alpha values used in the simulations.
        results_exhaustive_terminal (np.ndarray): The exhaustive terminal results as an array.
    """
    percentile_statistics = []

    # Iterate over all convergence points
    for index, convergence_point in enumerate(convergence_points_list):
        truncated_convergence_points = truncate_sublists(convergence_point, 7)
        truncated_array = np.array(truncated_convergence_points)

        # Calculate mean and std deviation across simulations at each episode
        episode_means = np.mean(truncated_array, axis=0)
        episode_stds = np.std(truncated_array, axis=0)

        # Convert the mean values to percentile scores
        episode_percentiles = [
            100 - stats.percentileofscore(results_exhaustive_terminal, mean_value)
            for mean_value in episode_means
        ]

        # Calculate the standard deviation of percentile scores
        std_percentiles = []
        for i in range(len(episode_means)):
            upper_value = episode_means[i] + episode_stds[i]
            lower_value = episode_means[i] - episode_stds[i]
            upper_percentile = 100 - stats.percentileofscore(results_exhaustive_terminal, upper_value)
            lower_percentile = 100 - stats.percentileofscore(results_exhaustive_terminal, lower_value)
            std_percentiles.append((upper_percentile - lower_percentile) / 2)

        percentile_statistics.append((episode_percentiles, std_percentiles))

    # Plotting
    matplotlib.rc('font', family='Arial', size='28')

    plt.figure(figsize=(10, 6))
    for index, (mean_percentiles, std_percentiles) in enumerate(percentile_statistics):
        x_values = list(range(len(mean_percentiles)))

        plt.plot(
            x_values,
            mean_percentiles,
            label=f'$\\alpha$: {alpha_values[index]}',
            linewidth=3
        )
        plt.fill_between(
            x_values,
            np.array(mean_percentiles) - np.array(std_percentiles),
            np.array(mean_percentiles) + np.array(std_percentiles),
            alpha=0.2
        )

    plt.xlabel('Episode')
    plt.ylabel('Percentile Score [%]')
    plt.ylim([98, 100])
    plt.legend(loc='lower right', fontsize='20')
    plt.show()


def plot_percentile_results_2(
    convergence_points_list: List[List[List[float]]],
    alpha_values: List[float],
    results_exhaustive_terminal: np.ndarray
):
    """
    Alternative method to plot percentile scores over episodes for different alpha values.

    Args:
        convergence_points_list (List[List[List[float]]]): List of convergence points for each alpha value.
        alpha_values (List[float]): List of alpha values used in the simulations.
        results_exhaustive_terminal (np.ndarray): The exhaustive terminal results as an array.
    """
    percentile_statistics = []

    for index, convergence_point in enumerate(convergence_points_list):
        truncated_convergence_points = truncate_sublists(convergence_point, 6)

        # Calculate episode percentiles
        episode_percentiles = []
        for episode_data in zip(*truncated_convergence_points):
            episode_scores = [
                100 - stats.percentileofscore(results_exhaustive_terminal, value)
                for value in episode_data
            ]
            episode_percentiles.append(episode_scores)

        # Calculate mean and std deviation of percentiles
        mean_percentiles = np.mean(episode_percentiles, axis=1)
        std_percentiles = np.std(episode_percentiles, axis=1)

        percentile_statistics.append((mean_percentiles, std_percentiles))

    # Plotting
    matplotlib.rc('font', family='Arial', size='28')

    plt.figure(figsize=(10, 6))
    for index, (mean_percentiles, std_percentiles) in enumerate(percentile_statistics):
        x_values = list(range(len(mean_percentiles)))

        plt.plot(
            x_values,
            mean_percentiles,
            label=f'$\\alpha$: {alpha_values[index]}',
            linewidth=3
        )
        plt.fill_between(
            x_values,
            mean_percentiles - std_percentiles,
            mean_percentiles + std_percentiles,
            alpha=0.2
        )

    plt.xlabel('Episode')
    plt.ylabel('Percentile Score [%]')
    plt.ylim([98, 100])
    plt.xlim([0, 1000])
    plt.legend(loc='lower right', fontsize='20')
    plt.show()
