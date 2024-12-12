# main.py

import time
from core import mcts 
from env import EnvTruss
from configurations import CONFIGURATIONS
from plot import plot_rewards, plot_min_results

def main():

    # PARAMETERS
    config_name = 'Ororbia_2'  # Choose the configuration you want to use
    config = CONFIGURATIONS[config_name]
    env = EnvTruss(config)
    # renderer = Renderer(env)
    
    start_time = time.time()
    
    num_eps = 1000
    alpha = 0.5
    attr_to_minimize = "max_displacement"  # 'strain_energy' or 'max_displacement'
    optimal_d = 0
    beta = 1.0  # This is only relevant in the 'mix-max' selection strategy
    
    """ SELECTION STRATEGY
    'UCT-normal'        - This is the basic strategy UCB for Trees
    'UCT-mixmax'        - Weights the max result found with 'beta' - [Jacobson 2014]
    'UCT-gabriel'       - Includes max term and standard deviation
    """
    select_strategy = 'UCT-mixmax'  # Choose your selection strategy
    
    # Run the MCTS algorithm
    root_node, min_node, results_episode, min_results_episode, fem_counter = mcts(
        env, env.initial_state, num_eps, alpha, attr_to_minimize, optimal_d, beta, select_strategy
    )
    
    elapsed_time = time.time() - start_time
    
    # ------------    RESULTS   --------------
    print("\n---- RESULTS ----")
    print(f"MCTS algorithm elapsed time: {elapsed_time:.4f} seconds")
    print(f"Number of FEM solves: {env.fem_counter}")
    print(f"Final Displacement: {min_node.max_displacement:.4f}")
    
    # Render the final truss structure
    env.render_node(min_node, title="Final Truss Structure")
    
    # Plot results
    plot_rewards(results_episode)
    plot_min_results(min_results_episode)
    
    # Print the highest value best path
    root_node.print_highest_value_best_path(env)


if __name__ == "__main__":
    main()