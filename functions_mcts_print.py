# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:34:44 2023

@author: gabri
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def chosenPath(env, node, attr_to_minimize="strain_energy"):
    """
    Follows the most visited path from the given node down the tree,
    collecting states along the way.
    """
    print("--- CHOSEN PATH ---")
    
    chosen_nodes_path = [node.state]

    while node.children:  
        # Select the child with the most visits
        node = max(node.children, key=lambda c: c.visits)
        chosen_nodes_path.append(node)

        if node.terminal:
            break
    
    chosen_node_attr_value = getattr(node, attr_to_minimize)
    print(f"chosen path gives {attr_to_minimize} = {chosen_node_attr_value:.4f}")
    values = node.returnValues()
    
    return chosen_nodes_path, values, node




    
    

def plotRewards(rewards_all_episodes):
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

def plotMinResults(rewards_all_episodes):
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