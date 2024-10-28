# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:34:44 2023

@author: gabri
"""
import math
import matplotlib.pyplot as plt
import numpy as np

from functions import calculate_reward, convert_state_to_set


def trace_back_and_print(env, node):
    if not node:
        print("Node is None")
        return
    
    displacement_values = []
    state_values = []
    node_values = []
    
    # Iteratively move up the tree until the root node is reached
    current_node = node
    
    while current_node is not None:
        displacement_values.append(current_node.maxDisplacement)
        state_values.append(current_node.state)
        node_values.append(current_node)
        current_node = current_node.parent

    # Since we've traced from terminal to root, reverse the list to print from root to terminal
    displacement_values.reverse()
    state_values.reverse()
    node_values.reverse()

    for i, node in enumerate(node_values):
        label = r'$s_{{{}}}$'.format(i)  # Using .format() to insert the subscript
        env.render0_black(node.state, label)
        print(node.maxDisplacement)


    
    return node_values


def minNode(env, nodeEpisode, attr_to_minimize="strain_energy"):
    
    # Using getattr to dynamically access the attribute to minimize
    min_node = min(nodeEpisode, key=lambda node: getattr(node, attr_to_minimize))
    
    print(f"--- MIN {attr_to_minimize.upper()} FOR ALL SIMULATIONS ---")
    min_node_attr_value = getattr(min_node, attr_to_minimize)
    print(f"min node gives {attr_to_minimize} = {min_node_attr_value:.4f}")
    
    
    values = min_node.returnValues()

    return values, min_node


def highestRewardNode(env, nodeEpisode, attr_to_minimize="maxDisplacement"):
    
    hr_node = max(nodeEpisode, key=lambda node: getattr(node, 'reward'))
    
    print(f"--- MIN {attr_to_minimize.upper()} FOR ALL SIMULATIONS ---")
    min_node_attr_value = getattr(hr_node, attr_to_minimize)
    print(f"min node gives {attr_to_minimize} = {min_node_attr_value:.4f}")
    
    values = hr_node.returnValues()

    return values, hr_node



def findNode(env, all_nodes, target_state):
    """
    Searches through a list of all nodes and returns the node with a state that matches the given state.
    
    """
    target_state_set = convert_state_to_set(target_state)
    
    for node in all_nodes:
        node_set = convert_state_to_set(node.state)
        if node_set == target_state_set:
            return node
    return False


def minDisplacementPhase1(env, nodeEpisode):
    """
    Searches every node in nodeEpisode, traces back up the tree from each node to find its 
    particular displacementMax value when parentTakenAction == 32. 
    Finds the node with the minimum of this value and saves it.
    """
    min_displacement = float('inf')
    min_node = None

    for node in nodeEpisode:
        current_node = node
        while current_node is not None:
            if hasattr(current_node, 'parentTakenAction') and current_node.parentTakenAction["n"] == 32:
                if current_node.displacementMax < min_displacement:
                    min_displacement = current_node.displacementMax
                    min_node = node  # Saving the node from nodeEpisode
                break
            current_node = current_node.parent

    if min_node is not None:
        print("--- MIN DISPLACEMENT FOR PARENT TAKEN ACTION 32 ---")
        minDisplacementReward = calculate_reward(env, min_node)
        node_values = trace_back_and_print(env, min_node)
        print(f"Min displacement at action 32 = {min_displacement}")
        print(f"And final reward = {minDisplacementReward}")
        # env.render0(min_node.state)
        return node_values
    else:
        print("No node with parentTakenAction == 32 found")
        return None


" ---- PLOTTING THE UCB FORMULA ---- "

def plotUCB():
    # Define the range of x values
    x_values = range(1, 8)  # no. of parent visits
    
    # Create the plot
    for x in x_values:
        results = []
        for y in range(1, x + 1):  # no. of child visits limited by parent visits
            result = math.sqrt(2 * math.log(x) / y)
            results.append(result)
        plt.plot([y for y in range(1, x + 1)], results, label=f'parent visits = {x}')
    
    plt.grid(which='both', axis='both', linestyle='-', color='gray', linewidth=0.5)
    
    plt.xticks(x_values)
    plt.yticks(np.arange(0, max(results) + 1, 0.5))  # Adjust y-axis ticks if necessary
    
    plt.xlabel('child visits')
    plt.ylabel('Result of equation')
    plt.title('Plot of sqrt(log(parent visits) / child visits)')
    plt.legend()
    plt.show()


" ---- PLOTTING THE UCB FORMULA ---- "
def plotUCB_2():
    y_values = range(1, 8)
    
    # Create the plot
    for y in y_values:
        results = []
        for x in range(y, 8):  # parent visits starting from the current child visit number
            result = math.sqrt(2 * math.log(x) / y)
            results.append(result)
        plt.plot(range(y, 8), results, label=f'child visits = {y}')
    
    plt.grid(which='both', axis='both', linestyle='-', color='gray', linewidth=0.5)
    
    plt.xticks(range(1, 8))  # x-axis represents parent visits
    plt.yticks(np.arange(0, max(results) + 1, 0.5))  # Adjust y-axis ticks if necessary
    
    plt.xlabel('parent visits')
    plt.ylabel('Result of equation')
    plt.title('Plot of sqrt(2 * log(parent visits) / child visits)')
    plt.legend()
    plt.show()
