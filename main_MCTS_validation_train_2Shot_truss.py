
" ------------   IMPORT STATEMENTS  -------------- "
import time
from pympler import asizeof
import json 
from core_MCTS_truss                  import mcts


" ------------  TRUSS -------------- "
from env_truss                  import env_truss
from functions_MCTS_print_truss import minNode
from functions_MCTS_print       import chosenPath


example = "Ororbia_4"
env                 = env_truss(example)
attr_to_minimize    = "maxDisplacement"  #'strain_energy' or 'maxDisplacement'

attribute_list = ['maxDisplacement']




" ------------   VALIDATION CODE  -------------- "

convergence_points  = []
alpha_values        = []

total_simulations   = 10
episodes_length     = 10000


for j in range(1):
    
    alpha       = j * 0.2 + 0.5
    alpha       = round(alpha, 1)
    alpha_values.append(alpha)
    
    beta = 0.0
    
    convergence_points_alpha_initial    = []
    convergence_points_alpha_final    = []
    
    
    for i in range(total_simulations):
        
        " MCTS "
        _, min_node, resultsEpisode, minResultsEpisode, maxRewardsEpisode, FEM_counter = mcts(env, env.initial_state, episodes_length, 0.1, attr_to_minimize, 0, beta)
        # convergence_points_alpha_initial.append(minResultsEpisode)
        
        min_d = minResultsEpisode[-1]
        
        print("")
        print(f"Episode {i}, for alpha value: {alpha:.3f}, minimizing {attr_to_minimize}")
        
        _, min_node, resultsEpisode, minResultsEpisode, maxRewardsEpisode, FEM_counter = mcts(env, env.initial_state, episodes_length, alpha,  attr_to_minimize, min_d, beta)
        
        
        convergence_points_alpha_final.append(minResultsEpisode)
        
        # env.render0(chosen_node.state)
    
    
    # convergence_path_name_initial = f"{example}_NEW_INITIAL_convergence_{attr_to_minimize}_{alpha}_{total_simulations}_{episodes_length}"
    convergence_path_name_final = f"{example}_2SHOT_convergence_{attr_to_minimize}_{alpha}_{beta}_{total_simulations}_{episodes_length}"

    # with open(f'validation/{convergence_path_name_initial}.json', 'w') as json_file:
    #     json.dump(convergence_points_alpha_initial, json_file)
    
    with open(f'validation/{convergence_path_name_final}.json', 'w') as json_file:
        json.dump(convergence_points_alpha_final, json_file)
        
        