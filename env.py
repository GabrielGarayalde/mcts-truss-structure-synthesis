#env_truss.py

import numpy as np
from copy import deepcopy
from configurations import (
    CantileverConfig,
    LShapeConfig,
    Ororbia1Config,
    Ororbia2Config,
    Ororbia3Config,
    Ororbia4Config,
    Ororbia5Config,
    Ororbia7Config,
)

from rendering import Renderer  # Import Renderer class


class EnvTruss:
    time_counter = 0
    fem_counter = 0

    def __init__(self, config):
        self.config = config
        # MATERIAL ATTRIBUTES #
        self.area = config.area
        self.emod = config.emod
        self.density = config.density

        # DOMAIN ATTRIBUTES #
        self.x = config.x
        self.y = config.y
        self.xm = config.xm
        self.yn = config.yn
        self.initial_state = np.array(config.initial_state)
        self.passive_nodes = config.passive_nodes
        self.max_volume = config.max_volume

        # Additional attributes
        self.node_sections = config.node_sections
        self.optimal_displacement = config.optimal_displacement
        self.optimal_states = config.optimal_states
        self.max_states = config.max_states
        self.optimal_strain_energy = config.optimal_strain_energy
        self.displacement_direction = config.displacement_direction
        self.construction_type = config.construction_type
        self.max_element_length = config.max_element_length
        
        # MESH
        self.x_spacing = int(self.x / (self.yn - 1)) if self.yn > 1 else self.x
        self.y_spacing = int(self.y / (self.xm - 1)) if self.xm > 1 else self.y
        self.grid = self._generate_mesh()
        self.truss_nodes_initial = self._create_truss_nodes()
        self.truss_nodes = deepcopy(self.truss_nodes_initial)
        self.truss_nodes_list = list(range(len(self.truss_nodes_initial)))

        # Initialize Renderer
        self.renderer = Renderer(self)  # Pass the EnvTruss instance to Renderer
        
    def _generate_mesh(self):
        """Generates the mesh grid for the truss structure."""
        nodes = []
        for j in range(0, self.xm * self.y_spacing, self.y_spacing):
            for k in range(0, self.yn * self.x_spacing, self.x_spacing):
                nodes.append([k, j])
        return nodes

    def _create_truss_nodes(self):
        """Creates the truss nodes and sets boundary conditions."""
        truss_nodes = []
        node_id = 0
        for i in range(len(self.grid)):
            node = {
                "x": self.grid[i][0],
                "y": self.grid[i][1],
                "ID": node_id,
                "freeDOF_x": True,
                "freeDOF_y": True,
                "force_x": 0,
                "force_y": 0
            }
            truss_nodes.append(node)
            node_id += 1

        # Set boundary conditions based on the config type
        self._set_boundary_conditions(truss_nodes)

        return truss_nodes

    def _set_boundary_conditions(self, truss_nodes):
        """Sets boundary conditions based on the configuration."""
        yn = self.yn
        xm = self.xm

        # Default boundary condition
        truss_nodes[0]['freeDOF_x'] = False
        truss_nodes[0]['freeDOF_y'] = False

        if isinstance(self.config, CantileverConfig):
            truss_nodes[yn]['freeDOF_x'] = False
            truss_nodes[yn]['freeDOF_y'] = False
            truss_nodes[2 * yn]['freeDOF_x'] = False
            truss_nodes[2 * yn]['freeDOF_y'] = False

        elif isinstance(self.config, LShapeConfig):
            index1 = yn * (xm - 1)
            index2 = yn * (xm - 1) + 2
            truss_nodes[index1]['freeDOF_x'] = False
            truss_nodes[index1]['freeDOF_y'] = False
            truss_nodes[index2]['freeDOF_x'] = False
            truss_nodes[index2]['freeDOF_y'] = False
            truss_nodes[8]['force_y'] = -10

        elif isinstance(self.config, Ororbia1Config):
            truss_nodes[2]['freeDOF_x'] = False
            truss_nodes[2]['freeDOF_y'] = False
            truss_nodes[11]['force_x'] = 10

        elif isinstance(self.config, Ororbia2Config):
            truss_nodes[2]['freeDOF_x'] = False
            truss_nodes[2]['freeDOF_y'] = False
            truss_nodes[8]['force_x'] = 10
            truss_nodes[14]['force_x'] = 10

        elif isinstance(self.config, Ororbia3Config):
            truss_nodes[4]['freeDOF_x'] = False
            truss_nodes[4]['freeDOF_y'] = False
            truss_nodes[24]['force_x'] = 10
            truss_nodes[24]['force_y'] = 10

        elif isinstance(self.config, Ororbia4Config):
            truss_nodes[9]['freeDOF_x'] = False
            truss_nodes[9]['freeDOF_y'] = False
            truss_nodes[27]['freeDOF_x'] = False
            truss_nodes[27]['freeDOF_y'] = False
            truss_nodes[26]['force_y'] = -10
        
        elif isinstance(self.config, Ororbia5Config):
            truss_nodes[4]['freeDOF_x'] = False
            truss_nodes[4]['freeDOF_y'] = False
            truss_nodes[2]['force_y'] = -10
        
        elif isinstance(self.config, Ororbia7Config):
            truss_nodes[6]['freeDOF_x'] = False
            truss_nodes[6]['freeDOF_y'] = False
            truss_nodes[48]['force_x'] = 10
            truss_nodes[48]['force_y'] = 10
            
    def render(self, state, title=None, stressMaxList=None, show_labels=False, label_rotation=0):
        """
        Renders the truss structure by delegating to the Renderer.

        Args:
            state (List[List[int]]): The truss state to render.
            title (str, optional): Title of the plot.
            stressMaxList (List[float], optional): List of stress values for each element.
            show_labels (bool, optional): Whether to show labels on elements.
            label_rotation (float, optional): Rotation angle for labels.
        """
        self.renderer.render(state, title, stressMaxList, show_labels, label_rotation)

    def render_node(self, node, title=None, show_labels=False, label_rotation=0):
            """
            Renders the elements connected to a specific node using the node's state.
    
            Args:
                node (dict): The node dictionary containing a 'state' key with connected elements.
                title (str, optional): Title of the plot.
                show_labels (bool, optional): Whether to show labels on elements.
                label_rotation (float, optional): Rotation angle for labels.
            """
    
            # Delegate rendering to Renderer without stress list
            self.renderer.render(
                state=node.state,
                title=title,
                stressMaxList=None,  # No stress list provided
                show_labels=show_labels,
                label_rotation=label_rotation
            )