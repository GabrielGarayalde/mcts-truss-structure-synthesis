# rendering.py

import matplotlib.pyplot as plt

class Renderer:
    """
    Renderer class for visualizing truss structures.
    """
    def __init__(self, env):
        """
        Initializes the renderer with a given environment.

        Args:
            env (EnvTruss): The environment object.
        """
        self.env = env

    def render(self, state, title=None, stressMaxList=None, show_labels=False, label_rotation=0):
        """
        Renders the truss structure with optional stress visualization and labels.

        Args:
            state (List[List[int]]): The truss state to render.
            title (str, optional): Title of the plot.
            stressMaxList (List[float], optional): List of stress values for each element.
            show_labels (bool, optional): Whether to show labels on elements.
            label_rotation (float, optional): Rotation angle for labels.
        """
        plt.figure(figsize=(10, 6))

        # If stressMaxList is provided, set up color mapping
        if stressMaxList is not None:
            colormap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=min(stressMaxList), vmax=max(stressMaxList))
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
        else:
            colormap = None

        for index, element in enumerate(state):
            node1 = self.env.truss_nodes[element[0]]
            node2 = self.env.truss_nodes[element[1]]

            # Determine color based on stress
            if stressMaxList is not None:
                color = colormap(norm(stressMaxList[index]))
                plt.plot([node1['x'], node2['x']], [node1['y'], node2['y']], color=color)
            else:
                plt.plot([node1['x'], node2['x']], [node1['y'], node2['y']], 'k-')  # 'k-' specifies black color and solid line

            # Add labels if required
            if show_labels:
                if stressMaxList is not None:
                    label = f'{stressMaxList[index]:.1f}'
                else:
                    label = str(index)
                mid_x = (node1['x'] + node2['x']) / 2
                mid_y = (node1['y'] + node2['y']) / 2
                plt.text(mid_x, mid_y, label, fontsize=8, color='black', rotation=label_rotation)

        # Scatter plot to show the grid nodes in grey
        plt.scatter(*zip(*self.env.grid), s=32, color='grey')

        # Gather all node positions from the elements
        all_nodes = set()
        for element in state:
            all_nodes.add(element[0])  # Add start node of element
            all_nodes.add(element[1])  # Add end node of element

        # Convert node ids to positions for plotting
        node_positions = [self.env.truss_nodes[node_id] for node_id in all_nodes]
        node_x, node_y = zip(*[(node['x'], node['y']) for node in node_positions])

        # Scatter plot to show the connecting grid nodes in black
        plt.scatter(node_x, node_y, s=32, color='black')

        if stressMaxList is not None:
            # Create a colorbar to indicate stress values
            plt.colorbar(sm, label='Stress')

        if title:
            plt.title(title, fontdict={'family': 'Arial', 'size': 24}, pad=20)

        # Set the aspect of the plot to be equal, ensuring an evenly spaced grid
        plt.axis('equal')

        # Remove box and numbers around the edges
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.xticks([])  # Remove x-axis tick marks
        plt.yticks([])  # Remove y-axis tick marks

        plt.show()

    def render_node(self, node, title=None, **kwargs):
        """
        Renders a specific node's state.

        Args:
            node (Node): The node object to render.
            title (str, optional): Title of the plot.
            **kwargs: Additional keyword arguments for the render method.
        """
        self.render(node.state, title, **kwargs)
