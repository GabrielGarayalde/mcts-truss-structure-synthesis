# functions.py

import math
import numpy as np


def is_game_over(env, node, run_type='normal'):
    """
    Determines if the game is over based on the node's state and the environment.

    Args:
        env: The environment object containing configuration and state.
        node: The current node in the search tree.
        run_type (str): The type of run ('normal' or 'exhaustive').

    Returns:
        bool: True if the game is over, False otherwise.
    """
    if run_type == 'normal':
        # Check if the node is terminal
        if node.terminal:
            return True

    # Terminal state condition
    if env.config.construction_type == "static":
        if node.depth == env.max_states:
            node.terminal = True
            return True
    elif env.config.construction_type == "progressive":
        yn = env.yn
        terminal_node = yn - 1
        for element in node.state:
            if terminal_node in element:
                node.terminal = True
                return True  # Found the terminal value

    if not node.children:
        return False
    
    elif all(not child.allowed for child in node.children):
        node.terminal = True
        return True

    return False


def calculate_reward(env, node, disp_optimal=0, reward_type="max_displacement"):
    """
    Calculates the reward for a given node.

    Args:
        env: The environment object containing configuration and state.
        node: The current node in the search tree.
        disp_optimal (float): The target optimal displacement (default is 0).
        reward_type (str): The type of reward to calculate ('max_displacement' or 'strain_energy').

    Returns:
        float: The calculated reward.
    """
    volume = node.volume

    if reward_type == "max_displacement":
        disp_current = node.max_displacement
        disp_base = env.initial_attribute  # Set this for progressive construction examples.

        if volume > env.max_volume or disp_current > disp_base:
            return 0
        else:
            # Reward between [0, 1] based on displacement improvement
            return (disp_base - disp_current) / (disp_base - disp_optimal)
    elif reward_type == "strain_energy":
        # Implement strain energy reward calculation if needed
        return 0
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def move_check(env, node):
    """
    Checks if a move is valid based on action, element length, and volume constraints.

    Args:
        env: The environment object containing configuration and state.
        node: The current node in the search tree.

    Returns:
        Tuple[List[List[int]], bool]: The new state and a boolean indicating if the move is valid.
    """
    node.failed_conditions = ""

    action_check_bool, new_state = action_check(env, node)
    if hasattr(env, 'max_element_length'):
        element_length_check_bool = element_length_check(new_state, env)
    else:
        element_length_check_bool = True
    volume_check_bool = volume_check(env, new_state)

    # Check and record failed conditions
    if not action_check_bool:
        node.failed_conditions += "Action Check Failed. "
    if not element_length_check_bool:
        node.failed_conditions += "Element Length Check Failed. "
    if not volume_check_bool:
        node.failed_conditions += "Volume Check Failed. "

    # Check if all conditions are met before proceeding
    if not (action_check_bool and element_length_check_bool and volume_check_bool):
        return new_state, False

    # Clear failed conditions if all checks passed
    node.failed_conditions = ""
    return new_state, True


def calculate_means_stds(sublists):
    """
    Calculates the mean and standard deviation for each index across sublists.

    Args:
        sublists (List[List[float]]): A list of sublists containing numerical values.

    Returns:
        Tuple[List[float], List[float]]: Lists of means and standard deviations.
    """
    transposed = list(zip(*sublists))
    means = [np.mean(index_values) for index_values in transposed]
    stds = [np.std(index_values) for index_values in transposed]
    return means, stds


def allowable_nodes_list(env, node):
    """
    Returns a list of allowable nodes for a given node.

    Args:
        env: The environment object containing configuration and state.
        node: The current node in the search tree.

    Returns:
        List[int]: A list of allowable node IDs.
    """
    active_nodes, inactive_nodes = active_nodes_list(env, node.state)
    allowable_nodes = [n for n in inactive_nodes if n not in env.passive_nodes]
    return allowable_nodes


def state_actions_list_create(env, node):
    """
    Creates a list of all allowable state actions for a given state.

    Args:
        env: The environment object containing configuration and state.
        node: The current node in the search tree.

    Returns:
        np.ndarray: An array of possible actions.
    """
    state_actions = []
    allowable_nodes = allowable_nodes_list(env, node)

    for element_index in range(len(node.state)):
        for node_id in allowable_nodes:
            for operator in range(2):
                action = [element_index, node_id, operator]
                state_actions.append(action)

    return np.array(state_actions)


def line(node1_index, node2_index, nodes):
    """
    Calculates the slope and y-intercept of a line between two nodes.

    Args:
        node1_index (int): Index of the first node.
        node2_index (int): Index of the second node.
        nodes (List[Dict]): List of node dictionaries.

    Returns:
        Tuple[Optional[float], Optional[float]]: Slope and y-intercept of the line.
    """
    node1 = nodes[node1_index]
    node2 = nodes[node2_index]

    x1, y1 = node1['x'], node1['y']
    x2, y2 = node2['x'], node2['y']

    if x1 == x2:
        return None, None  # Vertical line
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y2 - slope * x2
        return slope, intercept


def compute_distance(node1_index, node2_index, nodes):
    """
    Computes the Euclidean distance between two nodes.

    Args:
        node1_index (int): Index of the first node.
        node2_index (int): Index of the second node.
        nodes (List[Dict]): List of node dictionaries.

    Returns:
        float: The distance between the two nodes.
    """
    node1 = nodes[node1_index]
    node2 = nodes[node2_index]

    x1, y1 = node1['x'], node1['y']
    x2, y2 = node2['x'], node2['y']

    return math.hypot(x2 - x1, y2 - y1)


def element_length_check(state, env):
    """
    Checks if all elements in the state are shorter than the maximum element length.

    Args:
        state (List[List[int]]): The current state representing active elements.
        env: The environment object containing configuration and state.

    Returns:
        bool: True if all elements are within the maximum length, False otherwise.
    """
    nodes = env.truss_nodes

    for element in state:
        elem_length = compute_distance(element[0], element[1], nodes)
        if not env.max_element_length:
            return True
        elif elem_length > env.max_element_length:
            return False
    return True


def contains_nodes(node1_index, node2_index, nodes):
    """
    Finds nodes contained within the rectangle formed by two nodes.

    Args:
        node1_index (int): Index of the first node.
        node2_index (int): Index of the second node.
        nodes (List[Dict]): List of node dictionaries.

    Returns:
        List[int]: List of node IDs contained within the rectangle.
    """
    node1 = nodes[node1_index]
    node2 = nodes[node2_index]

    min_x = min(node1['x'], node2['x'])
    max_x = max(node1['x'], node2['x'])
    min_y = min(node1['y'], node2['y'])
    max_y = max(node1['y'], node2['y'])

    contained_nodes = []

    for node in nodes:
        x, y = node['x'], node['y']
        if min_x <= x <= max_x and min_y <= y <= max_y and node['ID'] not in [node1_index, node2_index]:
            contained_nodes.append(node['ID'])

    return contained_nodes


def active_nodes_list(env, state):
    """
    Returns lists of active and inactive node IDs based on the current state.

    Args:
        env: The environment object containing configuration and state.
        state (List[List[int]]): The current state representing active elements.

    Returns:
        Tuple[List[int], List[int]]: Lists of active and inactive node IDs.
    """
    all_nodes = env.truss_nodes_list
    active_nodes = set()

    for element in state:
        active_nodes.update(element)

    active_nodes = sorted(active_nodes)
    inactive_nodes = [node for node in all_nodes if node not in active_nodes]

    return active_nodes, inactive_nodes


def compute_volume(env, elements):
    """
    Computes the total volume of the truss structure.

    Args:
        env: The environment object containing configuration and state.
        elements (List[List[int]]): List of elements in the truss.

    Returns:
        float: The total volume of the truss.
    """
    area = env.area
    nodes = env.truss_nodes
    total_volume = 0.0

    for element in elements:
        length = compute_distance(element[0], element[1], nodes)
        element_volume = length * area
        total_volume += element_volume

    return total_volume


def volume_check(env, state):
    """
    Checks if the volume of the truss is within the maximum allowed volume.

    Args:
        env: The environment object containing configuration and state.
        state (List[List[int]]): The current state representing active elements.

    Returns:
        bool: True if the volume is within the limit, False otherwise.
    """
    return compute_volume(env, state) <= env.max_volume


def ccw(a_idx, b_idx, c_idx, grid):
    """
    Checks if three points are listed in a counter-clockwise order.

    Args:
        a_idx (int): Index of point A.
        b_idx (int): Index of point B.
        c_idx (int): Index of point C.
        grid (List[Tuple[float, float]]): List of point coordinates.

    Returns:
        bool: True if points are in counter-clockwise order.
    """
    ax, ay = grid[a_idx]
    bx, by = grid[b_idx]
    cx, cy = grid[c_idx]
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def line_intersect_check(a_idx, b_idx, c_idx, d_idx, grid):
    """
    Checks if line segment AB intersects with line segment CD.

    Args:
        a_idx (int): Index of point A.
        b_idx (int): Index of point B.
        c_idx (int): Index of point C.
        d_idx (int): Index of point D.
        grid (List[Tuple[float, float]]): List of point coordinates.

    Returns:
        bool: True if lines intersect, False otherwise.
    """
    if len({a_idx, b_idx, c_idx, d_idx}) < 4:
        return False
    return (ccw(a_idx, c_idx, d_idx, grid) != ccw(b_idx, c_idx, d_idx, grid) and
            ccw(a_idx, b_idx, c_idx, grid) != ccw(a_idx, b_idx, d_idx, grid))


def point_intersect_check(node_idx, element, nodes):
    """
    Checks if a node lies on a given element (line segment).

    Args:
        node_idx (int): Index of the node to check.
        element (List[int]): The element represented by two node indices.
        nodes (List[Dict]): List of node dictionaries.

    Returns:
        bool: True if the node lies on the element, False otherwise.
    """
    node = nodes[node_idx]
    contained_nodes = contains_nodes(element[0], element[1], nodes)

    if node_idx in contained_nodes:
        slope, intercept = line(element[0], element[1], nodes)
        if slope is None:
            # Vertical line
            return True
        elif abs(node['y'] - (slope * node['x'] + intercept)) <= 1e-10:
            return True
    return False


def calculate_area(x1, y1, x2, y2, x3, y3):
    """
    Calculates the area of a triangle formed by three points.

    Args:
        x1, y1 (float): Coordinates of the first point.
        x2, y2 (float): Coordinates of the second point.
        x3, y3 (float): Coordinates of the third point.

    Returns:
        float: The area of the triangle.
    """
    return abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0)


def is_point_inside_triangle(nodes, i1, i2, i3, p):
    """
    Checks if a point lies inside a triangle formed by three other points.

    Args:
        nodes (List[Dict]): List of node dictionaries.
        i1, i2, i3 (int): Indices of the triangle's vertices.
        p (int): Index of the point to check.

    Returns:
        bool: True if the point lies inside the triangle, False otherwise.
    """
    A = nodes[i1]
    B = nodes[i2]
    C = nodes[i3]
    P = nodes[p]

    area_abc = calculate_area(A['x'], A['y'], B['x'], B['y'], C['x'], C['y'])
    area_pab = calculate_area(P['x'], P['y'], A['x'], A['y'], B['x'], B['y'])
    area_pbc = calculate_area(P['x'], P['y'], B['x'], B['y'], C['x'], C['y'])
    area_pac = calculate_area(P['x'], P['y'], A['x'], A['y'], C['x'], C['y'])

    return abs(area_abc - (area_pab + area_pbc + area_pac)) < 1e-10


def inside_check(nodes, trial_edge, active_nodes, trial_node, state, env):
    """
    Determines if a trial node is inside, outside, or on the line of a trial edge.

    Args:
        nodes (List[Dict]): List of node dictionaries.
        trial_edge (List[int]): The edge being tested.
        active_nodes (List[int]): List of active node indices.
        trial_node (int): Index of the trial node.
        state (List[List[int]]): The current state representing active elements.
        env: The environment object containing configuration and state.

    Returns:
        str: "inside", "outside", or "on the line".
    """
    # Check if trial_node is on the trial_edge
    if point_intersect_check(trial_node, trial_edge, nodes):
        return "on the line"

    # Iterate through all active nodes to form a triangle with trial_edge
    for third_point in active_nodes:
        if third_point in trial_edge:
            continue

        intersection_found = False
        # Check for intersection with existing elements
        for existing_element in state:
            if (line_intersect_check(trial_edge[0], third_point, existing_element[0], existing_element[1], env.grid) or
                    line_intersect_check(trial_edge[1], third_point, existing_element[0], existing_element[1], env.grid)):
                intersection_found = True
                break

        if intersection_found:
            continue

        # Check if trial_node is inside the triangle
        if not is_point_inside_triangle(nodes, trial_edge[0], trial_edge[1], third_point, trial_node):
            return "outside"

    return "inside"


def convert_state_to_set(state):
    """
    Converts a state (list of elements) to a set of sorted tuples for comparison.

    Args:
        state (List[List[int]]): The state to convert.

    Returns:
        Set[Tuple[int, int]]: A set of sorted element tuples.
    """
    return set(tuple(sorted(sublist)) for sublist in state)


def percentage_optimal_states(chosen_nodes, optimal_states, total_simulations):
    """
    Calculates the percentage of optimal states found in simulations.

    Args:
        chosen_nodes (List[List[Dict]]): Nodes chosen in simulations.
        optimal_states (List[List[List[int]]]): Known optimal states.
        total_simulations (int): Total number of simulations run.

    Returns:
        List[float]: Percentages of optimal states per alpha value.
    """
    optimal_state_sets = [convert_state_to_set(opt_state) for opt_state in optimal_states]
    percentages = []

    for nodes_alpha in chosen_nodes:
        count = 0
        for node in nodes_alpha:
            node_state_set = convert_state_to_set(node['state'])
            if any(node_state_set == opt_state_set for opt_state_set in optimal_state_sets):
                count += 1
        percentage = (count / total_simulations) * 100
        percentages.append(percentage)

    return percentages


def find_sublist_index(number, node_sections):
    """
    Finds the index of the sublist containing a given number.

    Args:
        number (int): The number to search for.
        node_sections (List[List[int]]): A list of sublists.

    Returns:
        Optional[int]: The index of the sublist containing the number, or None if not found.
    """
    for index, sublist in enumerate(node_sections):
        if number in sublist:
            return index
    return None


def action_check(env, node):
    """
    Checks whether an action can be applied to the current node without violating any constraints.

    Args:
        env: The environment object containing truss configurations and parameters.
        node: The current node in the search tree.

    Returns:
        Tuple[bool, List[List[int]]]: A tuple where the first element is a boolean indicating if the action is allowed,
        and the second element is the new state after applying the action.
    """
    # Get active nodes from the parent state
    active_nodes, _ = active_nodes_list(env, node.parent.state)

    # Copy the parent state to create a new state
    new_state = node.parent.state.copy().tolist()

    # Extract action details from the node
    trial_e_index = node.parent_taken_action[0]
    trial_e = new_state[trial_e_index]
    trial_n = node.parent_taken_action[1]
    trial_op = node.parent_taken_action[2]

    # Check whether the trial node is inside, outside, or on the line of the trial element
    result = inside_check(env.truss_nodes, trial_e, active_nodes, trial_n, new_state, env)

    # Perform the action based on the operation type
    if trial_op == 0:
        # Remove the trial element for T operation
        del new_state[trial_e_index]

    # Append new elements connecting the trial node to the ends of the trial element
    new_state.extend([[trial_n, trial_e[0]], [trial_n, trial_e[1]]])

    # Initialize action failure flag
    action_fail = False

    # Check if the proposed lines intersect with existing lines
    action_fail = any(
        line_intersect_check(trial_n, trial_e[0], elem[0], elem[1], env.grid) or
        line_intersect_check(trial_n, trial_e[1], elem[0], elem[1], env.grid)
        for elem in new_state
    )

    # Check if the trial node lies on any existing elements
    if not action_fail:
        action_fail = any(
            point_intersect_check(trial_n, elem, env.truss_nodes)
            for elem in new_state
        )

    # Check if proposed lines pass over active nodes
    if not action_fail:
        action_fail = any(
            point_intersect_check(active_node, [trial_n, trial_e[0]], env.truss_nodes) or
            point_intersect_check(active_node, [trial_n, trial_e[1]], env.truss_nodes)
            for active_node in active_nodes
        )

    # If any of the checks failed, return False
    if action_fail:
        return False, new_state

    # Iterate through active nodes to connect to the trial node
    for trial_third_point in active_nodes:
        if trial_third_point in trial_e:
            continue  # Skip nodes already in the trial element

        # Flag to determine if we should skip adding the new element
        skip_adding = False

        # Check for intersections with existing elements
        skip_adding = any(
            line_intersect_check(trial_n, trial_third_point, elem[0], elem[1], env.grid)
            for elem in new_state
        )

        # Check if any active node lies on the proposed line
        if not skip_adding:
            skip_adding = any(
                point_intersect_check(active_node, [trial_n, trial_third_point], env.truss_nodes)
                for active_node in active_nodes
            )

        # Add the new element if all checks pass
        if not skip_adding:
            new_state.append([trial_n, trial_third_point])

    # Check if the original trial element intersects with any element in the new state
    element_intersect = any(
        line_intersect_check(trial_e[0], trial_e[1], elem[0], elem[1], env.grid)
        for elem in new_state
    )

    # Decide whether the action is allowed based on the operation and result
    if trial_op == 0 and result == 'outside':
        if element_intersect:
            return True, new_state
        else:
            return False, new_state
    else:
        return True, new_state



def stiffness_matrix_assemble(env, truss_nodes, state, self_weight=False):
    """
    Assembles the global stiffness matrix K and force vector F for the truss structure.

    Args:
        env: The environment object containing configuration and state.
        truss_nodes (List[Dict]): List of node dictionaries.
        state (List[List[int]]): Current state representing active elements.
        self_weight (bool): Whether to include self-weight in nodal loads.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[int]]: Stiffness matrix K, force vector F, and remaining DOFs.
    """
    if isinstance(state, np.ndarray):
        state = state.tolist()

    nodes = truss_nodes
    area = env.area
    emod = env.emod
    density = env.density

    active_nodes, _ = active_nodes_list(env, state)
    nnodes = len(active_nodes)

    K = np.zeros((2 * nnodes, 2 * nnodes))
    F = np.zeros((2 * nnodes, 1))

    for element in state:
        node1 = nodes[element[0]]
        node2 = nodes[element[1]]

        length = compute_distance(element[0], element[1], nodes)

        # Calculate nodal load if required
        if self_weight:
            nodal_load = -length * area * density * 9.81 / 2  # Self-weight
            try:
                idx1 = active_nodes.index(element[0])
                idx2 = active_nodes.index(element[1])
                F[2 * idx1 + 1, 0] += nodal_load
                F[2 * idx2 + 1, 0] += nodal_load
            except ValueError:
                pass

        # Assemble local stiffness matrix
        klocal = np.array([[1, 0, -1, 0],
                           [0, 0, 0, 0],
                           [-1, 0, 1, 0],
                           [0, 0, 0, 0]]) * area * emod / length

        ca = (node2['x'] - node1['x']) / length
        sa = (node2['y'] - node1['y']) / length
        rot = np.array([[ca, -sa, 0, 0],
                        [sa, ca, 0, 0],
                        [0, 0, ca, -sa],
                        [0, 0, sa, ca]])
        klocal = rot @ klocal @ rot.T

        try:
            temp = [2 * active_nodes.index(element[0]) + i for i in range(2)] + \
                   [2 * active_nodes.index(element[1]) + i for i in range(2)]
        except ValueError:
            continue

        # Populate global stiffness matrix K
        for i in range(4):
            for j in range(4):
                K[temp[i], temp[j]] += klocal[i, j]

    # Apply external forces from nodes
    for idx, node_idx in enumerate(active_nodes):
        node = nodes[node_idx]
        if node.get('force_x', 0):
            F[2 * idx, 0] += node['force_x']
        if node.get('force_y', 0):
            F[2 * idx + 1, 0] += node['force_y']

    # Define all DOFs
    dofs = [2 * idx + dof for idx in range(nnodes) for dof in range(2)]

    # Identify restrained DOFs
    dofs_bf = []
    for idx, node_idx in enumerate(active_nodes):
        node = nodes[node_idx]
        if not node.get('freeDOF_x', True):
            dofs_bf.append(2 * idx)
        if not node.get('freeDOF_y', True):
            dofs_bf.append(2 * idx + 1)

    # Remove restrained DOFs
    dofs_delete = [dofs.index(dof) for dof in dofs_bf if dof in dofs]
    K = np.delete(K, dofs_delete, axis=0)
    K = np.delete(K, dofs_delete, axis=1)
    F = np.delete(F, dofs_delete, axis=0)

    # Remaining DOFs
    dofs_remain = [dof for idx, dof in enumerate(dofs) if idx not in dofs_delete]

    return K, F, dofs_remain


def conditioning_check(K, F):
    """
    Checks the conditioning of the stiffness matrix K.

    Args:
        K (np.ndarray): Stiffness matrix.
        F (np.ndarray): Force vector.

    Returns:
        bool: True if the matrix is well-conditioned, False otherwise.
    """
    eigenvalues = np.linalg.eigvals(K)
    if abs(np.min(eigenvalues)) < 1e-10 or abs(np.max(eigenvalues) / np.min(eigenvalues)) > 5e4:
        return False
    return True





def calculate_max_displacement(direction, U):
    """
    Calculates the maximum displacement in the specified direction.

    Args:
        direction (str): The direction ('x', 'y', or 'x_and_y').
        U (np.ndarray): Displacement vector.

    Returns:
        float: The maximum displacement.
    """
    
    U = U.flatten()

    if direction == "x":
        displacements = [abs(U[i]) for i in range(0, len(U), 2)]
        return max(displacements)
    elif direction == "y":
        displacements = [abs(U[i]) for i in range(1, len(U), 2)]
        return max(displacements)
    elif direction == "x_and_y":
        x_displacements = [abs(U[i]) for i in range(0, len(U), 2)]
        y_displacements = [abs(U[i]) for i in range(1, len(U), 2)]
        total_displacements = [math.hypot(x, y) for x, y in zip(x_displacements, y_displacements)]
        return max(total_displacements)
    else:
        raise ValueError("Direction must be 'x', 'y', or 'x_and_y'.")
