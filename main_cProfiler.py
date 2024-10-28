# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 11:27:48 2024

@author: gabri
"""

import cProfile
import time
from core_MCTS_truss import mcts
from functions_MCTS_print import plotRewards, plotMinResults
from env_truss import env_truss

# ------------    MAIN CODE   --------------

def main():
    start_time = time.time()  # Start timing the entire script

    # PARAMETERS
    env                 = env_truss("bridge")
    num_eps             = 1000
    alpha               = 0.3
    attr_to_minimize    = "maxDisplacement"  # 'strain_energy' or 'maxDisplacement'
    optimal_d           = 0
    beta                = 0.0  # Relevant only in the 'mix-max' selection strategy

    select_strategy     = 'UCT-normal'   # Selection strategy

    # MCTS function
    root_node, min_node, resultsEpisode, minResultsEpisode, maxRewardsEpisode, FEM_counter = mcts(
        env, env.initial_state, num_eps, alpha, attr_to_minimize, optimal_d, beta, select_strategy
    )

    # Render results
    env.render(min_node)

    # Plot results
    plotRewards(resultsEpisode)
    plotMinResults(minResultsEpisode)
    plotMinResults(maxRewardsEpisode)

    # Print children of root node (if applicable)
    root_node.print_children()

    # Measure total elapsed time
    elapsed_total = time.time() - start_time
    print("\n---- TOTAL RESULTS ----")
    print(f"Total script elapsed time: {elapsed_total:.4f} seconds")
    print(f"Number of FEM solves: {env.U_solve_counter}")
    print(f"Final Displacement: {min_node.maxDisplacement:.4f}")

# Profile the main function using cProfile
cProfile.run('main()', 'profile_output.prof')

import pstats

# Load the profiling data from the file
p = pstats.Stats('profile_output.prof')

# Sort the results by cumulative time (the total time spent in the function and its sub-functions)
# and print the top 20 functions.
p.strip_dirs().sort_stats('cumulative').print_stats(50)
