import random
import time
from classes import Node
from functions import is_game_over, calculate_reward
import copy

def mcts(env, initial_state, num_eps, alpha, attr_to_minimize, optimal_d, beta, select_strategy):
    results_episode = []
    min_node = None
    min_results_episode = []

    root_node = Node(env)
    root_node.populate_root_node(env, initial_state)
    env.initial_attribute = root_node.max_displacement

    currentMin = float('inf')
    env.fem_counter = 0

    selection_time = 0
    expansion_time = 0
    simulation_time = 0
    backpropagation_time = 0

    for episode in range(num_eps):
        node = root_node
        env.phaseState = 0

        # --- SELECTION ---
        start_selection = time.time()
        while node.children and not is_game_over(env, node):
            if any(child.allowed is None for child in node.children):
                break
            node = node.select_child(alpha, beta, num_eps, episode, select_strategy)
        selection_time += time.time() - start_selection

        # --- EXPANSION ---
        start_expansion = time.time()
        if not is_game_over(env, node):
            if not node.children:
                node.generate_children(env)
                node.populate_children(env)
        expansion_time += time.time() - start_expansion

        # --- SIMULATION ---
        start_simulation = time.time()
        while not is_game_over(env, node):
            if not node.children:
                node.generate_children(env)
                node.populate_children(env)
            selected_child = random.choice(node.children)
            
            if selected_child.allowed:
                node = selected_child
                # env.render(node)
        simulation_time += time.time() - start_simulation

        # if iteration % 50 == 0:
        #     print(iteration)
        #     env.render_node(node)
        #     print("max displacement", node.max_displacement)
        # Calculate reward
        reward = calculate_reward(env, node, optimal_d)
        # print(reward)
        nodeResult = getattr(node, attr_to_minimize)
        if nodeResult < currentMin:
            min_node = copy.deepcopy(node)
            currentMin = nodeResult
            # min_node = Node(env)
            # min_node.state = node.state
            # min_node.max_displacement = node.max_displacement

        results_episode.append(reward)
        min_results_episode.append(currentMin)

        # --- BACKPROPAGATION ---
        start_backpropagation = time.time()
        while node is not None:
            node.update(reward)
            node = node.parent
        backpropagation_time += time.time() - start_backpropagation

    fem_counter = env.fem_counter

    # Print or return the times for each stage
    print(f"Total Selection Time: {selection_time:.4f} seconds")
    print(f"Total Expansion Time: {expansion_time:.4f} seconds")
    print(f"Total Simulation Time: {simulation_time:.4f} seconds")
    print(f"Total Backpropagation Time: {backpropagation_time:.4f} seconds")

    return root_node, min_node, results_episode, min_results_episode, fem_counter