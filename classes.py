import math
import random
import numpy as np

from functions import (
    state_actions_list_create,
    compute_volume,
    calculate_max_displacement,
    move_check,
    stiffness_matrix_assemble,
    conditioning_check,
)


class Node:
    """
    Node class for Monte Carlo Tree Search (MCTS) in truss optimization.

    Attributes:
        parent (Node): The parent node.
        parent_taken_action: The action taken from the parent to reach this node.
        children (list of Node): List of child nodes.
        state (np.ndarray): The state represented by this node.
        visits (int): Number of times this node has been visited.
        depth (int): Depth of this node in the tree.
        allowed (bool): Whether this node is allowed (e.g., satisfies constraints).
        terminal (bool): Whether this node is a terminal node.
        value (float): Cumulative value of this node.
        reward (float): Reward obtained at this node.
        average (float): Average value (value / visits).
        variance (float): Variance of the rewards.
        reward_list (list of float): List of rewards received.
        value_best (float): Best value obtained from this node.
        nodeID (int): Unique identifier for the node.
    """

    node_counter = 0  # Class variable to assign unique IDs to nodes

    def __init__(self, env, taken_action=None, parent=None):
        """
        Initializes a Node.

        Args:
            env: The environment.
            taken_action: The action taken from the parent to reach this node.
            parent (Node, optional): The parent node.
        """
        self.parent = parent
        self.parent_taken_action = taken_action

        self.children = []
        self.state = None

        self.visits = 0
        self.depth = 0 if parent is None else parent.depth + 1

        self.allowed = None
        self.terminal = False

        self.value = 0
        self.reward = 0
        self.average = 0
        self.variance = 0
        self.reward_list = []
        self.value_best = float('-inf')

        # Assign a unique node ID
        self.nodeID = Node.node_counter
        Node.node_counter += 1

    def populate_root_node(self, env, root_state):
        """
        Populates the root node with the initial state.

        Args:
            env: The environment.
            root_state (array-like): The initial state.
        """
        self.state = np.array(root_state)
        self.allowed = True
        self.populate_node(env)

    def populate_node(self, env):
        """
        Populates the node by computing necessary values like stiffness matrix, volume, etc.

        Args:
            env: The environment.
        """
        if self.allowed:
            # Determine the stiffness matrix assembly based on env configuration
            if env.config.construction_type == 'static':
                K, F, _ = stiffness_matrix_assemble(env, env.truss_nodes, self.state)
            elif env.config.construction_type == 'progressive':
                K, F, _ = stiffness_matrix_assemble(env, env.truss_nodes, self.state, self_weight=True)
            else:
                raise ValueError(f"Unknown construction type: {env.config.construction_type}")

            # Check if the system is well-conditioned
            self.allowed = conditioning_check(K, F)
            self.volume = compute_volume(env, self.state)

        if self.allowed:
            # Solve for displacements
            U = np.linalg.solve(K, F)
            env.fem_counter += 1

            # Calculate the maximum displacement
            self.max_displacement = calculate_max_displacement(env.displacement_direction, U)
            self.U = U  # Store displacement vector if needed
        else:
            # Node is not allowed, set terminal flag
            self.terminal = True

    def generate_children(self, env):
        """
        Generates child nodes from possible actions.

        Args:
            env: The environment.
        """
        state_actions_list = state_actions_list_create(env, self)

        for taken_action in state_actions_list:
            child_node = Node(env, taken_action, self)
            self.children.append(child_node)

    def populate_children(self, env):
        """
        Populates child nodes by checking moves and populating their states.

        Args:
            env: The environment.
        """
        for child in self.children:
            if child.allowed is None:
                new_state, allowed = move_check(env, child)
                child.state = np.array(new_state)
                child.allowed = allowed
                child.populate_node(env)

    def select_child(self, alpha, beta, iterations, i, select_strategy):
        """
        Selects a child node based on the specified selection strategy.

        Args:
            alpha (float): Exploration-exploitation balance parameter.
            beta (float): Additional parameter for certain strategies.
            iterations (int): Total number of iterations.
            i (int): Current iteration number.
            select_strategy (str): The selection strategy to use.

        Returns:
            Node: The selected child node, or None if no allowed children are available.
        """
        # Filter children to include only those where 'allowed' is True
        allowed_children = [child for child in self.children if child.allowed is True]

        unvisited_children = [child for child in allowed_children if child.visits == 0]
        if unvisited_children:
            return random.choice(unvisited_children)

        if allowed_children:
            if select_strategy == 'UCT-normal':
                # Original UCB formula
                return max(
                    allowed_children,
                    key=lambda c: (1 - alpha) * c.average + alpha * math.sqrt(2 * math.log(self.visits) / c.visits)
                )
            elif select_strategy == 'UCT-mixmax':
                # UCB - Mixmax modification
                return max(
                    allowed_children,
                    key=lambda c: (1 - alpha) * ((1 - beta) * (c.value / c.visits) + beta * c.value_best) +
                                  alpha * math.sqrt(2 * math.log(self.visits) / c.visits)
                )
            elif select_strategy == 'UCT-schadd':
                # UCT - Schadd 2008 modification
                return max(
                    allowed_children,
                    key=lambda c: (1 - alpha) * c.average +
                                  alpha * (math.sqrt(2 * math.log(self.visits) / c.visits) +
                                           math.sqrt(c.variance + 1 / self.visits))
                )
            elif select_strategy == 'UCT-gabriel':
                # UCT - Gabriel's modification
                return max(
                    allowed_children,
                    key=lambda c: (1 - alpha) * ((1 - beta) * c.average + beta * c.value_best) +
                                  alpha * (math.sqrt(c.variance) + math.sqrt(2 * math.log(self.visits) / c.visits))
                )
            else:
                raise ValueError(f"Invalid selection strategy '{select_strategy}'")
        else:
            # No allowed children
            return None

    def update(self, result):
        """
        Updates the node with the result of a simulation.

        Args:
            result (float): The result to update the node with.
        """
        self.visits += 1
        self.value += result
        self.reward = result  # Stores the last reward received

        self.average = self.value / self.visits
        self.reward_list.append(result)

        # Calculate variance
        if self.visits > 1:
            squared_diffs = [(x - self.average) ** 2 for x in self.reward_list]
            self.variance = sum(squared_diffs) / self.visits
        else:
            self.variance = 0

        if result > self.value_best:
            self.value_best = result

    def print_children(self, sort='average'):
        """
        Prints information about the node's children.

        Args:
            sort (str): The attribute to sort the children by. Defaults to 'average'.
        """
        # Filter children to include only those where 'allowed' is True
        allowed_children = [(index, child) for index, child in enumerate(self.children) if child.allowed is True]

        print(f"Visits: {self.visits}, # of allowed children: {len(allowed_children)}, "
              f"Average: {self.average:.5f}, Best: {self.value_best:.5f}, Variance: {self.variance:.5f}\n")

        # Validate the sort criteria
        valid_criteria = ['average', 'value_best', 'variance', 'visits']
        if sort not in valid_criteria:
            print(f"Invalid sort criteria '{sort}', defaulting to 'average'.")
            sort = 'average'

        # Sort the allowed children based on the selected criteria
        sorted_children = sorted(allowed_children, key=lambda x: getattr(x[1], sort), reverse=True)

        # Print details for each allowed child
        for order_index, (original_index, child) in enumerate(sorted_children, start=1):
            allowed_or_none_count = len([c for c in child.children if c.allowed in (True, None)])
            allowed_or_none_count_str = allowed_or_none_count if allowed_or_none_count > 0 else "None"
            print(f"{order_index}, Index: {original_index}, # child: {allowed_or_none_count_str}, "
                  f"Visits: {child.visits}, Parent Action: {child.parent_taken_action}, "
                  f"Average: {child.average:.5f}, Best: {child.value_best:.5f}, Variance: {child.variance:.5f}")

    def print_highest_value_best_path(self, env):
        """
        Prints and renders the path with the highest value_best from this node downwards.

        Args:
            env: The environment.
        """
        node = self
        print(f"Visits: {node.visits}, Best: {node.value_best:.8f}, "
              f"Average: {node.average:.8f}, Variance: {node.variance:.8f}")

        env.render(node.state)

        while node.children:
            # Filter children to include only those where 'allowed' is True
            allowed_children = [child for child in node.children if child.allowed is True]

            if not allowed_children:
                break

            # Find the child with the highest value_best attribute
            best_child = max(allowed_children, key=lambda child: child.value_best)
            best_child_index = node.children.index(best_child)

            # Print details of the best child
            print(f"Index: {best_child_index}, Visits: {best_child.visits}, "
                  f"Parent Action: {best_child.parent_taken_action}, Best: {best_child.value_best:.8f}, "
                  f"Average: {best_child.average:.8f}, Variance: {best_child.variance:.8f}")

            # Render the best child node
            env.render(best_child.state)

            # Move to the next node (best child)
            node = best_child

    def print_values(self):
        """
        Prints the node's values.
        """
        print(f"Node ID: {self.nodeID}")
        print(f"Visits: {self.visits}")
        print(f"Value: {self.value}")
        print(f"Reward: {self.reward}")
        print(f"Max Displacement: {getattr(self, 'max_displacement', 'N/A')}")

    def to_dict(self):
        """
        Converts the node to a dictionary for serialization.

        Returns:
            dict: A dictionary representation of the node.
        """
        return {
            "nodeID": self.nodeID,
            "parent_nodeID": self.parent.nodeID if self.parent else None,
            "parent_taken_action": self.parent_taken_action,
            "children_nodeIDs": [child.nodeID for child in self.children],
            "state": self.state.tolist() if isinstance(self.state, np.ndarray) else self.state,
            "value": self.value,
            "visits": self.visits,
            "depth": self.depth,
            "allowed": self.allowed,
            "terminal": self.terminal,
            "reward": self.reward,
            "U": self.U.tolist() if hasattr(self, 'U') and isinstance(self.U, np.ndarray) else getattr(self, 'U', None),
            "volume": getattr(self, "volume", None),
            "max_displacement": getattr(self, "max_displacement", None),
            "strain_energy": getattr(self, "strain_energy", None),
        }
