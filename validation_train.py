import time
import json
import os
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, NamedTuple
from decimal import Decimal
import numpy as np
from numpy.typing import NDArray

from core import mcts
from env import EnvTruss
from configurations import CONFIGURATIONS

# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SimulationConfig(NamedTuple):
    """Configuration parameters for MCTS simulation."""
    env: EnvTruss
    num_eps: int
    attr_to_minimize: str
    select_strategy: str
    total_sims: int
    alpha: float
    beta: float
    optimal_d: float

@dataclass
class SimulationResults:
    """Container for simulation results."""
    min_nodes: List[Any]
    elapsed_times: List[float]
    min_result_episodes: List[Any]
    FEM_counters: List[int]

class NodeState:
    """Container for node state information."""
    def __init__(self, state: NDArray[np.float64], max_displacement: float):
        self.state = state
        self.max_displacement = Decimal(str(max_displacement))  # Store as Decimal for full precision

    def to_dict(self) -> Dict[str, Any]:
        """Convert node state to dictionary format."""
        return {
            "state": self.state.tolist(),
            "max_displacement": str(self.max_displacement)  # Convert Decimal to string to preserve precision
        }

def run_single_simulation(
    config: SimulationConfig,
    simulation_index: int
) -> Optional[tuple[Any, float, Any, int]]:
    """
    Run a single MCTS simulation with given parameters.

    Args:
        config: SimulationConfig object containing all necessary parameters
        simulation_index: Index of current simulation

    Returns:
        Optional tuple containing (min_node, elapsed_time, min_result_episode, FEM_counter)
    """
    logger.info(
        f"Episode {simulation_index + 1}, for alpha value: {config.alpha:.3f}, "
        f"minimizing {config.attr_to_minimize}"
    )

    start_time = time.time()

    try:
        _, min_node, results_episode, min_result_episode, FEM_counter = mcts(
            env=config.env,
            initial_state=config.env.initial_state,
            num_eps=config.num_eps,
            alpha=config.alpha,
            attr_to_minimize=config.attr_to_minimize,
            optimal_d=config.optimal_d,
            beta=config.beta,
            select_strategy=config.select_strategy
        )

        elapsed_time = time.time() - start_time
        
        logger.info(f"Elapsed time: {elapsed_time:.4f} seconds")
        logger.info(f"Amount of FEM simulations: {FEM_counter}")

        try:
            min_node_attr_value = getattr(min_node, config.attr_to_minimize)
            logger.info(
                f"--- MIN {config.attr_to_minimize.upper()} FOR ALL SIMULATIONS ---\n"
                f"min node gives {config.attr_to_minimize} = {min_node_attr_value:.4f}"
            )
        except AttributeError as e:
            logger.error(f"Failed to access attribute {config.attr_to_minimize}: {e}")
            return None

        # Render the environment with the min node state
        try:
            config.env.render(min_node.state)
        except Exception as e:
            logger.error(f"Failed to render environment: {e}")
            # Continue execution as rendering failure is non-critical

        return min_node, elapsed_time, min_result_episode, FEM_counter

    except Exception as e:
        logger.error(f"MCTS simulation failed: {e}", exc_info=True)
        return None

def run_simulations(config: SimulationConfig) -> SimulationResults:
    """
    Run multiple MCTS simulations with given parameters.

    Args:
        config: SimulationConfig object containing all necessary parameters

    Returns:
        SimulationResults object containing all simulation results
    """
    results = SimulationResults([], [], [], [])

    for i in range(config.total_sims):
        simulation_result = run_single_simulation(config, i)
        
        if simulation_result:
            min_node, elapsed_time, min_result_episode, FEM_counter = simulation_result
            results.min_nodes.append(min_node)
            results.elapsed_times.append(elapsed_time)
            results.min_result_episodes.append(min_result_episode)
            results.FEM_counters.append(FEM_counter)

    return results

def save_results(
    example: str,
    attr_to_minimize: str,
    alpha: float,
    beta: float,
    total_sims: int,
    num_eps: int,
    results: SimulationResults,
    output_dir: str = "validation"
) -> None:
    """
    Save simulation results to a JSON file with full precision for numerical values.

    Args:
        example: The example name (e.g., 'bridge')
        attr_to_minimize: The attribute being minimized
        alpha: Alpha parameter
        beta: Beta parameter
        total_sims: Number of simulations
        num_eps: Number of episodes per simulation
        results: SimulationResults object containing all results
        output_dir: Base directory to save results
    """
    example_dir = os.path.join(output_dir, example)
    os.makedirs(example_dir, exist_ok=True)

    json_file_name = (
        f"{example}_results_{attr_to_minimize}_{alpha}_{beta}_{total_sims}_{num_eps}.json"
    )
    json_save_path = os.path.join(example_dir, json_file_name)

    # Process min_nodes with full precision
    min_nodes_processed = [
        NodeState(node.state, node.max_displacement).to_dict()
        for node in results.min_nodes
    ]

    consolidated_results = {
        "metadata": {
            "example": example,
            "attribute_to_minimize": attr_to_minimize,
            "alpha": str(Decimal(str(alpha))),  # Convert to Decimal for full precision
            "beta": str(Decimal(str(beta))),
            "total_simulations": total_sims,
            "episodes_per_simulation": num_eps,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        },
        "results": {
            "min_nodes": min_nodes_processed,
            "elapsed_times": results.elapsed_times,
            "min_result_episodes": results.min_result_episodes,
            "FEM_counters": results.FEM_counters
        }
    }

    try:
        with open(json_save_path, 'w') as json_file:
            json.dump(consolidated_results, json_file, indent=4)
        logger.info(f"Results successfully saved to {json_save_path}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}", exc_info=True)
        raise

def main() -> None:
    """Execute MCTS simulations with various parameters and save results."""
    # Configuration parameters
    config_name = 'Ororbia_1'
    config = CONFIGURATIONS[config_name]
    env = EnvTruss(config)
    
    simulation_params = {
        'num_eps': 50,
        'attr_to_minimize': "max_displacement",
        'select_strategy': 'UCT-normal',
        'total_sims': 2,
        'optimal_d': 0.0,
        'alpha_range': [0.1, 0.3, 0.5],
        'beta_range': [0.0, 0.2, 0.4]
    }

    for alpha in simulation_params['alpha_range']:
        for beta in simulation_params['beta_range']:
            logger.info(f"\nStarting simulations for alpha={alpha}, beta={beta}")

            sim_config = SimulationConfig(
                env=env,
                num_eps=simulation_params['num_eps'],
                attr_to_minimize=simulation_params['attr_to_minimize'],
                select_strategy=simulation_params['select_strategy'],
                total_sims=simulation_params['total_sims'],
                alpha=alpha,
                beta=beta,
                optimal_d=simulation_params['optimal_d']
            )

            results = run_simulations(sim_config)

            save_results(
                example=config_name,
                attr_to_minimize=simulation_params['attr_to_minimize'],
                alpha=alpha,
                beta=beta,
                total_sims=simulation_params['total_sims'],
                num_eps=simulation_params['num_eps'],
                results=results
            )

if __name__ == "__main__":
    main()