
" ------------   IMPORT STATEMENTS  -------------- "
import time
import json 
from core_MCTS_truss                 import mcts
import numpy as np

# from pympler import asizeof

# from memory_profiler import memory_usage
" ------------  TRUSS -------------- "
from env_truss                  import env_truss
from functions_MCTS_print_truss import minNode
from functions_MCTS_print       import chosenPath
from functions_mcts_truss_all        import is_game_over
from classes_MCTS_truss              import Node

" ------------  FLOOR -------------- "
# from functions_MCTS_floor            import is_game_over_exhaustive, calculateReward
# from classes_MCTS_floor              import Node


example = "bridge"
env                 = env_truss(example)
attr_to_minimize    = "maxDisplacement"  #'strain_energy' or 'maxDisplacement'

attribute_list = ['maxDisplacement']


" ---------  MCTS EXHAUSTIVE ----------- "
start_time          = time.time()


def node_to_dict(node, attribute, attribute_value):
    # Convert a node object to a dictionary
    return {
        "attribute": attribute,
        "attribute_value": attribute_value,
        "state": node.state.tolist(),
    }

def save_to_array(temporary_values):
    global all_nodes
    # Convert the list of temporary values to a NumPy array
    temp_array = np.array(temporary_values, dtype=np.float32)
    # Concatenate the temporary array with the main array
    all_nodes = np.concatenate((all_nodes, temp_array))

def collect_min_node_with_parents(node, attribute):
    """
    This function collects the current min node and all its parent nodes up to the root.
    """
    global min_nodes
    path_to_root = []

    # Traverse from the current node to the root
    while node is not None:
        attribute_value = getattr(node, attribute)
        path_to_root.append(node_to_dict(node, attribute, attribute_value))
        node = node.parent  # Move to the parent node

    min_nodes = list(reversed(path_to_root))

" ------------   MAIN CODE  -------------- "

all_nodes = np.empty((0,), dtype=np.float32)  # Initialize as an empty NumPy array
min_nodes = []
counter = 0  # Initialize the counter
temp_nodes = []  # Temporary list to collect node values before batching

# Create the root node
root_node = Node(env)
root_node.populate_root_node(env, env.initial_state)

min_attribute_value = float("inf")

def explore_node(node, partition_size=10):
    global counter, temp_nodes, min_attribute_value, min_nodes
    
    if is_game_over(env, node, run_type='exhaustive'):
        # env.render(node)
        counter += 1  
        if counter % 100 ==0:
            print(f"Explored {counter} terminal nodes...")
            # env.render(node)
        value_float32 = float(node.maxDisplacement)
        temp_nodes.append(value_float32)
        
        if len(temp_nodes) >= partition_size:
            save_to_array(temp_nodes)
            temp_nodes.clear()
        
        attribute_value = getattr(node, 'maxDisplacement')
        
        # If we find a new minimum node, collect it and its parents
        if attribute_value < min_attribute_value:
            min_attribute_value = attribute_value
            collect_min_node_with_parents(node, 'maxDisplacement')

    else:
        if not node.children:
            node.generate_children(env)
        for child in node.children:
            if child.allowed is None:
                child.populate_node(env)
                child.populate_node_attributes(env)
            if child.allowed:
                explore_node(child)
    
    if hasattr(node, 'children'):
        del node.children

explore_node(root_node)

# After exploring all nodes, save any remaining values in temp_nodes
if temp_nodes:
    save_to_array(temp_nodes)

# Save the final concatenated NumPy array to disk
def save_final_array(all_nodes, filename=f"validation/{example}/{example}_exhaustive_terminal.npy"):
    np.save(filename, all_nodes)

save_final_array(all_nodes)

mcts_elapsed_time = time.time() - start_time
min_value = min(all_nodes)
env.render0(min_nodes[-1]['state'], "Global Minimum configuration")

print(f"Total nodes explored with MCTS exhaustive: {counter}")
print(f"MCTS elapsed time: {mcts_elapsed_time:.4f}")
print(f"Min value for all nodes: {min_value}")

# Save the min_nodes array to a JSON file
with open(f'validation/{example}/{example}_exhaustive_terminal_min_nodes.json', 'w') as json_file:
    json.dump(min_nodes, json_file)
