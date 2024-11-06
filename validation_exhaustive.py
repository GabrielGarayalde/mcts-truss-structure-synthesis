import time
import pickle
import os
from core import mcts
import numpy as np

# Import necessary classes and functions
from functions import is_game_over
from classes import Node

from env import EnvTruss
from configurations import CONFIGURATIONS

# Initialize the environment and set parameters
example = "Ororbia_2"
config = CONFIGURATIONS[example]
env = EnvTruss(config)
attr_to_minimize = "max_displacement"

# Initialize variables for tracking
min_nodes = []
counter = 0
min_attribute_value = float("inf")

temp_nodes = []  # Temporary list to collect node dictionaries before batching

# Create the root node
root_node = Node(env)
root_node.populate_root_node(env, env.initial_state)

# Start timing
start_time = time.time()

def node_to_dict(node, attribute, attribute_value):
    """
    Converts a node object to a dictionary with the attribute name as the key.

    Args:
        node (Node): The node object.
        attribute (str): The attribute to minimize (e.g., 'max_displacement').
        attribute_value (float): The value of the attribute.

    Returns:
        dict: A dictionary with the attribute and state.
    """
    return {
        attribute: attribute_value,  # Store as float for Pickle
        "state": node.state.tolist(),
    }

def collect_min_node_with_parents(node, attribute):
    """
    Collects the current min node and all its parent nodes up to the root.

    Args:
        node (Node): The current node with the minimum attribute value.
        attribute (str): The attribute being minimized.
    """
    global min_nodes
    path_to_root = []

    # Traverse from the current node to the root
    while node is not None:
        attribute_value = getattr(node, attribute)
        path_to_root.append(node_to_dict(node, attribute, attribute_value))
        node = node.parent  # Move to the parent node

    # Reverse to have the path from root to min node
    min_nodes = list(reversed(path_to_root))

def save_to_pickle(temporary_values, filename):
    """
    Saves the temporary values to a Pickle file.

    Args:
        temporary_values (list): List of node dictionaries to save.
        filename (str): Path to the Pickle file.
    """
    with open(filename, 'ab') as f:
        pickle.dump(temporary_values, f)

def explore_node(node, partition_size=100):
    """
    Recursively explores the MCTS tree to collect all terminal nodes and identify the minimum node.

    Args:
        node (Node): The current node being explored.
        partition_size (int): Number of nodes to collect before saving to disk.
    """
    global counter, min_attribute_value, min_nodes, temp_nodes

    if is_game_over(env, node, run_type='exhaustive'):
        counter += 1
        if counter % 100 == 0:
            print(f"Explored {counter} terminal nodes...")

        # Create node dictionary
        attribute_value = getattr(node, 'max_displacement')
        node_dict = node_to_dict(node, 'max_displacement', attribute_value)
        temp_nodes.append(node_dict)

        # Save temp_nodes to file when it reaches partition_size
        if len(temp_nodes) >= partition_size:
            save_to_pickle(temp_nodes, nodes_filename)
            temp_nodes.clear()

        # Update min_nodes if necessary
        if attribute_value < min_attribute_value:
            min_attribute_value = attribute_value
            collect_min_node_with_parents(node, 'max_displacement')

    else:
        if not node.children:
            node.generate_children(env)
            node.populate_children(env)

        for child in node.children:
            if child.allowed is None:
                child.populate_node(env)
            if child.allowed:
                explore_node(child)

        # Clean up to save memory
        if hasattr(node, 'children'):
            del node.children

# Define the directory and filenames
output_dir = f"validation/{example}"
os.makedirs(output_dir, exist_ok=True)
nodes_filename = os.path.join(output_dir, f"{example}_all_nodes_exhaustive.pkl")
min_nodes_filename = os.path.join(output_dir, f"{example}_min_nodes_exhaustive.pkl")

# Initialize the Pickle file by clearing existing content
with open(nodes_filename, 'wb') as f:
    pass  # Creates or clears the file

# Start exploring the MCTS tree
explore_node(root_node)

# After exploring all nodes, save any remaining nodes in temp_nodes
if temp_nodes:
    save_to_pickle(temp_nodes, nodes_filename)
    temp_nodes.clear()

# Save the min_nodes to a separate Pickle file
with open(min_nodes_filename, 'wb') as f:
    pickle.dump(min_nodes, f)

# Render the minimal configuration if available
if min_nodes:
    env.render(min_nodes[-1]['state'], "Global Minimum Configuration")

# Print summary of the exploration
mcts_elapsed_time = time.time() - start_time
print(f"Total nodes explored with MCTS exhaustive: {counter}")
print(f"MCTS elapsed time: {mcts_elapsed_time:.4f} seconds")
print(f"Min value for all nodes: {min_attribute_value}")
