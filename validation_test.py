import numpy as np
import json
import os
import logging
from typing import Dict, Any

from env_truss import env_truss  # Adjust import based on your project structure

# Import plotting functions
from validation_plots import (
    plot_percentile_score,
    plot_fem_evaluations,
    plot_result
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_results(
    example: str,
    attr_to_minimize: str,
    alpha: float,
    beta: float,
    total_sims: int,
    num_eps: int,
    output_dir: str = "validation"
) -> Dict[str, Any]:
    # Same as your existing load_results function
    json_file_name = f"{example}_results_{attr_to_minimize}_{alpha}_{beta}_{total_sims}_{num_eps}.json"
    json_save_path = os.path.join(output_dir, example, json_file_name)
    try:
        with open(json_save_path, 'r') as json_file:
            data = json.load(json_file)
        logger.info(f"Results loaded from {json_save_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {json_save_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        return {}

def validation_test():
    example = 'Ororbia_1'
    env = env_truss(example)
    attr_to_minimize = 'max_displacement'
    beta = 0.0
    total_sims = 10
    num_eps = 1000
    alpha_values = [0.3]  # Add more alpha values if needed

    # Load exhaustive results
    results_exhaustive_terminal = np.load(f"validation/{example}/{example}_exhaustive_terminal.npy")
    min_value_exhaust = min(results_exhaustive_terminal)

    exhaustive_terminal_min_nodes_path = f"validation/{example}/{example}_exhaustive_terminal_min_nodes"
    with open(f'{exhaustive_terminal_min_nodes_path}.json', 'r') as json_file:
        min_node_exhaustive = json.load(json_file)
    min_state_exhaust = min_node_exhaustive[-1]['state']

    # Initialize lists
    min_nodes_list = []
    elapsed_times_list = []
    convergence_points_list = []
    FEM_counter_list = []

    for alpha in alpha_values:
        results = load_results(
            example=example,
            attr_to_minimize=attr_to_minimize,
            alpha=alpha,
            beta=beta,
            total_sims=total_sims,
            num_eps=num_eps,
            output_dir="validation"
        )
        if not results:
            logger.error("No results to process.")
            continue

        min_nodes = results.get("results", {}).get("min_nodes", [])
        elapsed_times = results.get("results", {}).get("elapsed_times", [])
        convergence_points = results.get("results", {}).get("min_result_episodes", [])
        FEM_counters = results.get("results", {}).get("FEM_counters", [])

        min_nodes_list.append(min_nodes)
        elapsed_times_list.append(elapsed_times)
        convergence_points_list.append(convergence_points)
        FEM_counter_list.append(FEM_counters)

        print(f"Alpha: {alpha}")
        print(f"Total Simulations: {len(min_nodes)}")
        if elapsed_times:
            avg_elapsed_time = sum(elapsed_times) / len(elapsed_times)
            print(f"Average Elapsed Time: {avg_elapsed_time:.4f} seconds")
        print(f"FEM Simulations per Run: {FEM_counters}\n")

    # Plotting
    plot_percentile_score(results_exhaustive_terminal, min_nodes_list, alpha_values)
    plot_fem_evaluations(alpha_values, FEM_counter_list)
    plot_result(convergence_points_list, alpha_values, min_value_exhaust)

if __name__ == "__main__":
    validation_test()
