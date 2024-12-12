import numpy as np
import json
import pickle
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

from env import EnvTruss
from configurations import CONFIGURATIONS
from validation_plots import (
    plot_percentile_score,
    plot_fem_evaluations,
    plot_min_result_episodes,
    plot_elapsed_times,
    plot_objective_ratio,
    plot_percentile_results,
    plot_percentage_optimal_nodes
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class ExhaustiveResults:
    """Container for exhaustive search results."""
    terminal_nodes: List[Dict[str, Any]]
    min_nodes: List[Dict[str, Any]]
    min_value: float

@dataclass
class MCTSResults:
    """Container for MCTS results."""
    min_nodes: List[Dict[str, Any]]
    elapsed_times: List[float]
    min_result_episodes: List[Any]
    fem_counters: List[int]

def load_pickle_file(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load and combine data from a pickle file that may contain multiple dumps.

    Args:
        filepath: Path to the pickle file

    Returns:
        Combined list of all dumped data
    """
    all_data = []
    try:
        with open(filepath, 'rb') as f:
            while True:
                try:
                    batch = pickle.load(f)
                    all_data.extend(batch)
                except EOFError:
                    break
        logger.info(f"Successfully loaded pickle file: {filepath}")
        return all_data
    except Exception as e:
        logger.error(f"Error loading pickle file {filepath}: {e}")
        raise

def load_exhaustive_results(example: str, base_dir: str = "validation") -> ExhaustiveResults:
    """
    Load results from exhaustive search.

    Args:
        example: Name of the example configuration
        base_dir: Base directory for validation results

    Returns:
        ExhaustiveResults object containing all exhaustive search data
    """
    example_dir = Path(base_dir) / example
    
    # Load all terminal nodes
    terminal_nodes_path = example_dir / f"{example}_all_nodes_exhaustive.pkl"
    terminal_nodes = load_pickle_file(terminal_nodes_path)
    
    # Extract max_displacement values
    terminal_values = [float(node['max_displacement']) for node in terminal_nodes]
    min_value = min(terminal_values)
    
    # Load minimum nodes path
    min_nodes_path = example_dir / f"{example}_min_nodes_exhaustive.pkl"
    min_nodes = load_pickle_file(min_nodes_path)
    
    logger.info(f"Loaded exhaustive results for {example}")
    logger.info(f"Number of terminal nodes: {len(terminal_nodes)}")
    logger.info(f"Minimum value found: {min_value}")
    
    return ExhaustiveResults(terminal_nodes, min_nodes, min_value)

def load_mcts_results(
    example: str,
    attr_to_minimize: str,
    alpha: float,
    beta: float,
    total_sims: int,
    num_eps: int,
    base_dir: str = "validation"
) -> Optional[MCTSResults]:
    """Load results from MCTS simulations."""
    json_file_name = f"{example}_results_{attr_to_minimize}_{alpha}_{beta}_{total_sims}_{num_eps}.json"
    json_path = Path(base_dir) / example / json_file_name
    
    try:
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        
        results = data.get("results", {})
        mcts_results = MCTSResults(
            min_nodes=results.get("min_nodes", []),
            elapsed_times=results.get("elapsed_times", []),
            min_result_episodes=results.get("min_result_episodes", []),
            fem_counters=results.get("FEM_counters", [])
        )
        
        logger.info(f"Successfully loaded MCTS results from {json_path}")
        return mcts_results
    
    except FileNotFoundError:
        logger.error(f"File not found: {json_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        return None

def print_simulation_summary(alpha: float, mcts_results: MCTSResults) -> None:
    """Print summary statistics for MCTS simulations."""
    print(f"\nAlpha: {alpha}")
    print(f"Total Simulations: {len(mcts_results.min_nodes)}")
    
    if mcts_results.elapsed_times:
        avg_time = sum(mcts_results.elapsed_times) / len(mcts_results.elapsed_times)
        print(f"Average Elapsed Time: {avg_time:.4f} seconds")
    
    print(f"FEM Simulations per Run: {mcts_results.fem_counters}")

def validation_test() -> None:
    """Execute validation tests comparing MCTS results with exhaustive search."""
    # Configuration
    example = 'Ororbia_1'
    config = CONFIGURATIONS[example]
    env = EnvTruss(config)
    
    test_params = {
        'attr_to_minimize': 'max_displacement',
        'beta': 0.0,
        'total_sims': 10,
        'num_eps': 1000,
        'alpha_values': [0.3]  # Add more alpha values if needed
    }
    
    # Load exhaustive search results
    exhaustive_results = load_exhaustive_results(example)
    
    # Initialize results containers
    all_mcts_results: List[MCTSResults] = []
    
    # Load MCTS results for each alpha value
    for alpha in test_params['alpha_values']:
        mcts_result = load_mcts_results(
            example=example,
            attr_to_minimize=test_params['attr_to_minimize'],
            alpha=alpha,
            beta=test_params['beta'],
            total_sims=test_params['total_sims'],
            num_eps=test_params['num_eps']
        )
        
        if mcts_result:
            all_mcts_results.append(mcts_result)
            print_simulation_summary(alpha, mcts_result)
    
    # Extract data for plotting
    min_nodes_list = [result.min_nodes for result in all_mcts_results]
    fem_counter_list = [result.fem_counters for result in all_mcts_results]
    min_result_episodes_list = [result.min_result_episodes for result in all_mcts_results]
    elapsed_times_list = [result.elapsed_times for result in all_mcts_results]
    
    # Extract values for plotting
    terminal_values = [float(node['max_displacement']) for node in exhaustive_results.terminal_nodes]
    
    # Generate all plots
    plot_percentile_score(terminal_values, min_nodes_list, test_params['alpha_values'])
    plot_fem_evaluations(test_params['alpha_values'], fem_counter_list)
    plot_elapsed_times(test_params['alpha_values'], elapsed_times_list)
    
    plot_min_result_episodes(
        min_result_episodes_list,
        test_params['alpha_values'],
        exhaustive_results.min_value
    )
    
    plot_objective_ratio(
        min_result_episodes_list,
        test_params['alpha_values'],
        example,
        exhaustive_results.min_value
    )
    
    plot_percentile_results(
        min_result_episodes_list,
        test_params['alpha_values'],
        np.array(terminal_values)
    )
    
    plot_percentage_optimal_nodes(
        min_nodes_list,
        exhaustive_results.min_nodes[-1],  # The terminal node from exhaustive search
        test_params['total_sims'],
        test_params['alpha_values'],

    )

if __name__ == "__main__":
    validation_test()