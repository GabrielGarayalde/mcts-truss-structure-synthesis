## Mastering Truss Structure Synthesis
The following code has been used in the paper: "MCTS: Mastering truss structure synthesis". It allows the user to train and test the code for different pre-defined configurations. The idea is to use tree search to synthesise optimal design configurations from initial seed states.

## Code Walk-through

### `main.py`
Used when performing a single run-through of the MCTS algorithm. This calls upon the **core** module which contains the main body of the tree search algorithm. In the main file one can select the configuration (or case study) type from a set of configurations defined in the **configuration** module. These case studies are referenced in the paper.

### `core.py`
This contains the main MCTS algorithm. The algorithm is divided into the main 4 steps (selection, expansion, simulation, backpropogation) as per the literature. The inputs are:
- `env`: this is the environment module which contains important info about the truss environment
- `initial_state`: this refers to the seed configuration which is initialized after ever episode
- `num_eps`: the total number of episodes the MCTS algorithm runs for
- `attr_to_minimize`: default set to `max_displacement`, but allows for customizing. One could optimize for `strain_energy` however the implementation for this is not complete
- `optimal_d`: This affects the convergence rate of the `calculate_reward` function. By default set to 0. If one knows the optimal configuration displacement (or close to), say by running a few initial trial runs, one might be able to speed up convergence
- `beta`: value between [0:1] this is for the modified UCT-formula which includes a "maximum" term. It gives the proportion weighting of the "maximum" term with respect to the "average" term. If set to 0, then defaults to the "normal" UCT-formula which is based on solely average returns
- `select_strategy`: This refers to the UCT-formula used for the selection strategy. "UCT-normal" is the normal strategy, "UCT-mixmax" includes the beta term

  ### `configurations.py`
  The configurations file contains all the parameters and constraints of the case study examples. Using the syntax as a template one could create new examples. Some of the parameters in each of the configuration classes are optional (like optimal states) but can be useful if we want to call it in another module. All of these parameters are then passed onto the Env module

  ### `env.py`
  The env file contains all the parameters for the enironment, as well as the configs. One must set the boundary conditions here, which means fixing the nodes for each example. Importantly it has the render functions, which means in any script (so long as the environment `env` has been passed or initialized), one can call `env.render` and the node is rendered.

  ### `classes.py`
  This is the `Node` class, a repeatable function that is called everytime a new instance of a Node in the tree is created. Importantly it has methods to  `generate_children`, `populate_children`, and `select_child`, the last selection method using the UCT-formula criterion to select.

  ### `functions.py`
  The main functions file. Contains the majority of the functions used in the MCTS algorithm, from checking if the game is over, to calculating the reward, checking the validity of a move, as well as many smaller sub-functions

  ### `plot.py`
  The main plotting file.

  ### `validation_train.py`
  This file is used to run many consecutive training runs of the `core.py` mcts algorithm. It saves the main information about these runs (the configurations found, as well as the time, fem simulations and results) into specific `.json` files which can then be loading in the `validation_test.py` file. The idea is to save many runs to get a statistically significant understanding of the parameters on the result.

### TO DO
- fix truncation issue in validation_test and validation_plot
- fix all the graphs
- find a simpler way to create passive node configs.
  
