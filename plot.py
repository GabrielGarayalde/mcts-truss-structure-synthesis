# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 18:16:48 2024

@author: gabri
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def plot_rewards(rewards_all_episodes):
# Calculate the average of every 10 values
    batch_value = 10
    averages = [np.mean(rewards_all_episodes[i:i+batch_value]) for i in range(0, len(rewards_all_episodes), batch_value)]
    
    # Generate the indices corresponding to the average values
    indices = range(0, len(rewards_all_episodes), batch_value)
    
    # Plot average values against indices
    plt.plot(indices, averages)
    
    # Set labels and title
    plt.xlabel('episodes')
    plt.ylabel('Accumulated episodic reward')
    plt.title('Plot of rewards (average of batches of 10) against episodes')
    
    # Display the plot
    plt.show()

def plot_min_results(rewards_all_episodes):
# Calculate the average of every 10 values
    batch_value = 1
    averages = [np.mean(rewards_all_episodes[i:i+batch_value]) for i in range(0, len(rewards_all_episodes), batch_value)]
    
    # Generate the indices corresponding to the average values
    indices = range(0, len(rewards_all_episodes), batch_value)
    
    # Plot average values against indices
    plt.plot(indices, averages)
    
    # Set labels and title
    plt.xlabel('Episodes')
    plt.ylabel('Min Result found')
    plt.title('Plot of min result found vs Episode')
    
    # Display the plot
    plt.show()

def histogram_plot(array, color, results_exhaustive):
    count, bins, ignored = plt.hist(array, bins=20, density=True, color=color, alpha=0.5, label='Min Node')
    mu_min, sigma_min   = stats.norm.fit(array)
    x_min, x_max        = plt.xlim()
    x_range             = np.linspace(x_min, x_max, 100)
    p_min               = stats.norm.pdf(x_range, mu_min, sigma_min)
    plt.plot(x_range, p_min, color, linewidth=2)
    
    # Drawing vertical lines at each bin edge and annotating them
    for bin_edge in bins:
        percentile = stats.percentileofscore(results_exhaustive, bin_edge)
        plt.axvline(bin_edge, color='gray', linestyle='dotted')
        plt.text(bin_edge, plt.ylim()[1] * 0.95, f'{(100-percentile):.2f}%', rotation=90, va='top', color='gray')



def percentile_plot(main_array, value, color):
    percentile = stats.percentileofscore(main_array, value)
    plt.axvline(value, color=color, linestyle='dashed', linewidth=2)
    plt.text(value, plt.ylim()[1] * 0.75, f'{(100-percentile):.2f}%', rotation=90, va='top', color=color, fontsize=12)

    print(f"{value:.4f} is {(100-percentile):.2f}%")


def plot_ratios(node):
    # Filter children to include only those where 'allowed' is True
    allowed_children = [(index, child) for index, child in enumerate(node.children) if child.allowed is True]
    
    print(f"Visits: {node.visits}, #child {len(allowed_children)}, Average: {node.average:.5f}, Best: {node.value_best:.5f}, Variance: {node.variance:.5f}")
    print("")

    # Prepare data for plotting
    child_counts = []
    visit_counts = []
    visit_ratios = []

    # Collect data
    for original_index, child in allowed_children:
        allowed_or_none_count = len([c for c in child.children if c.allowed in (True, None)])
        if allowed_or_none_count > 0:
            visit_ratio = child.visits / allowed_or_none_count
            visit_ratios.append((original_index, visit_ratio))
            child_counts.append((original_index, allowed_or_none_count))
            visit_counts.append((original_index, child.visits))

    # Sort by the number of children (from lowest to highest)
    child_counts.sort(key=lambda x: x[1])
    sorted_indices = [index for index, _ in child_counts]

    # Reorder visit_counts and visit_ratios based on sorted_indices
    visit_counts.sort(key=lambda x: sorted_indices.index(x[0]))
    visit_ratios.sort(key=lambda x: sorted_indices.index(x[0]))

    # Extract the sorted values for plotting
    indices, sorted_child_counts = zip(*child_counts)
    _, sorted_visit_counts = zip(*visit_counts)
    _, sorted_visit_ratios = zip(*visit_ratios)

    # Plot the number of children
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(sorted_child_counts)), sorted_child_counts, color='skyblue')
    plt.xlabel('Child Index')
    plt.ylabel('Number of Children')
    plt.title('Number of Children for Each Child')
    plt.show()

    # Plot the number of visits
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(sorted_visit_counts)), sorted_visit_counts, color='lightgreen')
    plt.xlabel('Child Index')
    plt.ylabel('Number of Visits')
    plt.title('Number of Visits for Each Child')
    plt.show()

    # Plot the ratio of visits to the number of children
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(sorted_visit_ratios)), sorted_visit_ratios, color='lightcoral')
    plt.xlabel('Child Index')
    plt.ylabel('Visit Ratio (Visits / Number of Children)')
    plt.title('Visit Ratio of Children')
    plt.show()

# Usage example
# plot_ratios(node)



# Usage example
# plot_ratios(node)



" INCLUDES THE BETA - WITH THE MINMAX FORMULA "
def calculate_uct_components(node, alpha, beta, select_strategy):
    """ Calculate the components of the UCT formula for a node based on the selection strategy. """
    if node.parent is None or node.visits == 0 or not node.allowed:
        return [0] * 3  # Return a list of zeros based on the maximum number of components needed
    
    if select_strategy == 'UCT-normal':
        exploitation = (1 - alpha) * (node.average)
        exploration = alpha * math.sqrt(2 * math.log(node.parent.visits) / node.visits)
        return exploitation, exploration, 0
    
    elif select_strategy == 'UCT-mixmax':
        exploitation_term1 = (1 - alpha) * (1 - beta) * (node.average)
        exploitation_term2 = (1 - alpha) * beta * node.value_best
        exploration = alpha * math.sqrt(2 * math.log(node.parent.visits) / node.visits)
        return exploitation_term1, exploitation_term2, exploration
    
    elif select_strategy == 'UCT-schadd':
        exploitation = (1 - alpha) * node.average
        exploration1 = alpha * math.sqrt(2 * math.log(node.parent.visits) / node.visits)
        exploration2 = alpha * (math.sqrt(node.variance + 1/node.visits))
        return exploitation, exploration1, exploration2
    
    elif select_strategy == 'UCT-gabriel':
        exploitation = (1 - alpha) * ((1-beta)*node.average + beta*node.value_best)
        exploration1 = alpha * (math.sqrt(node.variance))
        exploration2 = alpha * math.sqrt(2 * math.log(node.parent.visits) / node.visits)
        return exploitation, exploration1, exploration2
    
    return [0] * 3


import matplotlib

def plot_uct_values_at_depth(env, root, depth, alpha, beta, episode, select_strategy):
    matplotlib.rc('font', family='Arial', size=16)

    """Plots and saves UCT values of all nodes at a given depth with Mixmax modification."""
    level_nodes = get_nodes_at_depth(root, depth)
    term1_values = []
    term2_values = []
    term3_values = []
    
    for node in level_nodes:
        term1, term2, term3 = calculate_uct_components(node, alpha, beta, select_strategy)
        term1_values.append(term1)
        term2_values.append(term2)
        term3_values.append(term3)

    uct_values = [x + y + z for x, y, z in zip(term1_values, term2_values, term3_values)]
    max_uct_value = max(uct_values)
    max_uct_index = uct_values.index(max_uct_value)

    max_term1 = max(term1_values)
    max_term2 = max(term2_values)
    max_term3 = max(term3_values)

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))  # Adjusted to plot 4 subplots

    special_node_index = 136

    # Define colors
    term1_color = 'skyblue'  # Sky Blue
    term2_color = 'royalblue'  # Royal Blue
    term3_color = 'navy'  # Navy Blue

    light_green = '#90EE90'  # Light Green
    medium_sea_green = '#3CB371'  # Medium Sea Green
    dark_green = '#006400'  # Dark Green

    light_coral = '#F08080'  # Light Coral
    crimson = '#DC143C'  # Crimson
    dark_red = '#8B0000'  # Dark Red

    # Plot for Term 1
    for i, value in enumerate(term1_values):
        color = dark_green if i == special_node_index else dark_red if i == max_uct_index else term3_color
        axs[0].bar(i, value, color=color)
    axs[0].set_title('Term 1')
    axs[0].set_xlabel('Node Index')
    axs[0].set_ylabel('Value')

    # Plot for Term 2
    for i, value in enumerate(term2_values):
        color = medium_sea_green if i == special_node_index else crimson if i == max_uct_index else term2_color
        axs[1].bar(i, value, color=color)
    axs[1].set_title('Term 2')
    axs[1].set_xlabel('Node Index')
    axs[1].set_ylabel('Value')

    # Plot for Term 3
    for i, value in enumerate(term3_values):
        color = light_green if i == special_node_index else light_coral if i == max_uct_index else term1_color
        axs[2].bar(i, value, color=color)
    axs[2].set_title('Term 3')
    axs[2].set_xlabel('Node Index')
    axs[2].set_ylabel('Value')

    # Plot for UCT Values
    for i, (term1, term2, term3) in enumerate(zip(term1_values, term2_values, term3_values)):
        color1 = dark_green if i == special_node_index else dark_red if i == max_uct_index else term3_color
        color2 = medium_sea_green if i == special_node_index else crimson if i == max_uct_index else term2_color
        color3 = light_green if i == special_node_index else light_coral if i == max_uct_index else term1_color
        axs[3].bar(i, term1, color=color1)
        axs[3].bar(i, term2, bottom=term1, color=color2)
        axs[3].bar(i, term3, bottom=term1 + term2, color=color3)
    axs[3].set_title('UCT Values')
    axs[3].set_xlabel('Node Index')
    axs[3].set_ylabel('Value')

    plt.suptitle(f'Example 4 - UCT Values at Depth {depth} - Episode {episode}')
    
    # plt.savefig(f'gif/uct_plot_episode_{episode}.png')  # Save plot as PNG
    
    plt.show()
    
    plt.close()  # Close the figure to free memory
    
    # Identify and print the index of the max UCT value node and its components
    print(f"Max UCT Node Index: {max_uct_index}, (Exploitation: {term1_values[max_uct_index]:.4f}, Exploration: {term2_values[max_uct_index]:.4f}, UCT: {uct_values[max_uct_index]:.4f})")
