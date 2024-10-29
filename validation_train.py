# ------------   IMPORT STATEMENTS  -------------- 
import time
import json
import os
import logging
from typing import List, Dict, Any

from core import mcts
from env import EnvTruss
from configurations import CONFIGURATIONS

# ------------   SETUP LOGGING  -------------- 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------   VALIDATION CODE  -------------- 

def run_simulations(
    env: EnvTruss,
    num_eps: int,
    attr_to_minimize: str,
    select_strategy: str,
    total_sims: int,
    alpha: float,
    beta: float,
    optimal_d: float,
) -> Dict[str, List[Any]]:
    """
    Run multiple MCTS simulations with given parameters.

    Args:
        env (env_truss): The environment object.
        num_eps (int): Number of episodes per simulation.
        attr_to_minimize (str): Attribute to minimize ('strain_energy' or 'max_displacement').
        select_strategy (str): Selection strategy for MCTS.
        total_sims (int): Number of simulations to run.
        alpha (float): Alpha parameter for MCTS.
        beta (float): Beta parameter for MCTS.

    Returns:
        Dict[str, List[Any]]: Dictionary containing simulation results.
    """
    min_nodes: List[Dict[str, Any]] = []
    elapsed_times: List[float] = []
    min_result_episodes: List[Any] = []
    FEM_counters: List[int] = []

    for i in range(total_sims):
        logger.info(f"Episode {i + 1}, for alpha value: {alpha:.3f}, minimizing {attr_to_minimize}")

        start_time = time.time()

        try:
            # Run MCTS
            _, min_node, results_episode, min_result_episode, FEM_counter = mcts(
                env=env,
                initial_state=env.initial_state,
                num_eps=num_eps,
                alpha=alpha,
                attr_to_minimize=attr_to_minimize,
                optimal_d=optimal_d,
                beta=beta,
                select_strategy=select_strategy
            )
        except Exception as e:
            logger.error(f"Error during MCTS simulation: {e}")
            continue

        elapsed_time = time.time() - start_time
        elapsed_times.append(elapsed_time)

        logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")
        logger.info(f"Amount of FEM simulations: {FEM_counter}")

        try:
            min_node_attr_value = getattr(min_node, attr_to_minimize)
            logger.info(f"--- MIN {attr_to_minimize.upper()} FOR ALL SIMULATIONS ---")
            logger.info(f"min node gives {attr_to_minimize} = {min_node_attr_value:.4f}")
        except AttributeError as e:
            logger.error(f"Attribute error: {e}")
            min_node_attr_value = None

        # Render the environment with the min node state
        try:
            env.render(min_node.state)
        except Exception as e:
            logger.error(f"Error during rendering: {e}")

        # Collect results
        min_nodes.append(min_node)
        min_result_episodes.append(min_result_episode)
        FEM_counters.append(FEM_counter)

    return {
        "min_nodes": min_nodes,
        "elapsed_times": elapsed_times,
        "min_result_episodes": min_result_episodes,
        "FEM_counters": FEM_counters
    }

def save_results(
    example: str,
    attr_to_minimize: str,
    alpha: float,
    beta: float,
    total_sims: int,
    num_eps: int,
    results: Dict[str, List[Any]],
    output_dir: str = "validation"
) -> None:
    """
    Save simulation results to a single JSON file, including only node.state and node.max_displacement.

    Args:
        example (str): The example name (e.g., 'bridge').
        attr_to_minimize (str): The attribute being minimized.
        alpha (float): Alpha parameter.
        beta (float): Beta parameter.
        total_sims (int): Number of simulations.
        num_eps (int): Number of episodes per simulation.
        results (Dict[str, List[Any]]): The simulation results.
        output_dir (str, optional): Base directory to save results. Defaults to "validation".
    """
    # Create example-specific directory
    example_dir = os.path.join(output_dir, example)
    os.makedirs(example_dir, exist_ok=True)

    # Define file name
    json_file_name = f"{example}_results_{attr_to_minimize}_{alpha}_{beta}_{total_sims}_{num_eps}.json"

    # Define full path
    json_save_path = os.path.join(example_dir, json_file_name)

    # Process min_nodes to extract only state and max_displacement
    min_nodes_processed = [
        {
            "state": node.state.tolist(),
            "max_displacement": node.max_displacement
        }
        for node in results.get("min_nodes", [])
    ]

    # Consolidate all results into a single dictionary
    consolidated_results = {
        "metadata": {
            "example": example,
            "attribute_to_minimize": attr_to_minimize,
            "alpha": alpha,
            "beta": beta,
            "total_simulations": total_sims,
            "episodes_per_simulation": num_eps,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        },
        "results": {
            "min_nodes": min_nodes_processed,
            "elapsed_times": results.get("elapsed_times", []),
            "min_result_episodes": results.get("min_result_episodes", []),
            "FEM_counters": results.get("FEM_counters", [])
        }
    }

    try:
        # Save consolidated results to JSON file
        with open(json_save_path, 'w') as json_file:
            json.dump(consolidated_results, json_file, indent=4)
        logger.info(f"All results saved to {json_save_path}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")



def main():
    """
    Main function to execute the MCTS simulations and save results.
    """
    # PARAMETERS
    config_name = 'Ororbia_1'  # Choose the configuration you want to use
    config = CONFIGURATIONS[config_name]
    env = EnvTruss(config)
    num_eps = 50
    attr_to_minimize = "max_displacement"  # Options: 'strain_energy' or 'max_displacement'
    optimal_d = 0

    # SELECTION STRATEGY
    select_strategy = 'UCT-normal'  # Options: 'UCT-mixmax', 'UCT-gabriel', etc.
    total_sims = 2

    alpha_range = [0.1, 0.3, 0.5]
    beta_range = [0.0, 0.2, 0.4]
    
    # Loop over different alpha and beta values
    for alpha in alpha_range:
        for beta in beta_range:

            logger.info(f"\nStarting simulations for alpha={alpha}, beta={beta}")
    
            # Run simulations
            results = run_simulations(
                env=env,
                num_eps=num_eps,
                attr_to_minimize=attr_to_minimize,
                select_strategy=select_strategy,
                total_sims=total_sims,
                alpha=alpha,
                beta=beta,
                optimal_d=optimal_d
            )
    
            # Define path names and save results
            save_results(
                example=config_name,
                attr_to_minimize=attr_to_minimize,
                alpha=alpha,
                beta=beta,
                total_sims=total_sims,
                num_eps=num_eps,
                results=results,
                output_dir="validation"
            )

if __name__ == "__main__":
    main()
