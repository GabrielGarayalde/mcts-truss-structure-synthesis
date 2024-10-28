" ------------   IMPORT STATEMENTS  -------------- "
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import json
from functions_MCTS_print       import histogram_plot
from functions_mcts_truss_all   import percentage_optimal_states #, calculate_means_stds



" ------------  TRUSS -------------- "
from env_truss                  import env_truss

example                 = "Ororbia_4"
env                     = env_truss(example)
attr_to_minimize        = 'maxDisplacement'


" ---------  MCTS VALIDATION RUNS ----------- "
alpha_values            = [0.5]
beta                    = 0.0
total_simulations       = 10
episodes_length         = 10000


convergence_points_original           = []
convergence_points_1st_shot          = []
convergence_points_2nd_shot          = []


for alpha in alpha_values:
    
    convergence_points_alpha_initial    = []
    convergence_points_alpha_final    = []
    
    convergence_path_name_original = f"{example}_convergence_{attr_to_minimize}_{alpha}_{beta}_{total_simulations}_{episodes_length}"
    
    convergence_path_name_1st_shot = f"{example}_convergence_{attr_to_minimize}_0.1_{beta}_{total_simulations}_{episodes_length}"

    convergence_path_name_2nd_shot = f"{example}_2SHOT_convergence_{attr_to_minimize}_{alpha}_{beta}_{total_simulations}_{episodes_length}"

    with open(f'validation/{example}/{convergence_path_name_original}.json', 'r') as json_file:
        convergence_data_original = json.load(json_file)
    
    with open(f'validation/{example}/{convergence_path_name_1st_shot}.json', 'r') as json_file:
        convergence_data_1st_shot = json.load(json_file)
    
    with open(f'validation/{example}/{convergence_path_name_2nd_shot}.json', 'r') as json_file:
        convergence_data_2nd_shot = json.load(json_file)
    
    convergence_points_original.append(convergence_data_original)
    convergence_points_1st_shot.append(convergence_data_1st_shot)
    convergence_points_2nd_shot.append(convergence_data_2nd_shot)





" -------------  G R A P H S ------------- "


# Placeholder function to calculate means and standard deviations
def calculate_means_stds(sublists):
    means = np.mean(sublists, axis=0)
    stds = np.std(sublists, axis=0)
    return means, stds

min_attribute = env.optimalDisplacement

cutoff = 10000

for index, (sublists, sublists_1st, sublists_2nd) in enumerate(zip(convergence_points_original, convergence_points_1st_shot, convergence_points_2nd_shot)):
    plt.figure()
    means, stds = calculate_means_stds(sublists)
    means_1st, stds_1st = calculate_means_stds(sublists_1st)
    means_2nd, stds_2nd = calculate_means_stds(sublists_2nd)

    x_values = list(range(len(sublists[0])))

    if len(x_values) > cutoff:
        x_values = x_values[:cutoff]
        means, stds = means[:cutoff], stds[:cutoff]
        means_1st, stds_1st = means_1st[:cutoff], stds_1st[:cutoff]
        means_2nd, stds_2nd = means_2nd[:cutoff], stds_2nd[:cutoff]

    # Plot for original convergence points
    plt.plot(x_values, means, color='blue', label=f'Alpha {alpha_values[index]:.2f} Original')
    plt.fill_between(x_values, np.subtract(means, stds), np.add(means, stds), color='blue', alpha=0.2)

    # Plot for 1st shot convergence points
    plt.plot(x_values, means_1st, color='orange', linestyle='dashed', label=f'Alpha {alpha_values[index]:.2f} 1st Shot')
    plt.fill_between(x_values, np.subtract(means_1st, stds_1st), np.add(means_1st, stds_1st), color='orange', alpha=0.1)

    # Plot for 2nd shot convergence points
    plt.plot(x_values, means_2nd, color='green', label=f'Alpha {alpha_values[index]:.2f} 2nd Shot')
    plt.fill_between(x_values, np.subtract(means_2nd, stds_2nd), np.add(means_2nd, stds_2nd), color='green', alpha=0.1)

    # Add a horizontal line for the global minimum attribute value
    plt.axhline(y=min_attribute, color='red', linestyle='dashed', linewidth=2)
    plt.text(max(x_values)*0.1, min_attribute, f'Global Min: {min_attribute:.4f}', va='top', ha='left', color='red')

    plt.xlabel('Episode')
    plt.ylabel('Displacement')
    plt.title('Comparison of 2-Shot Results vs Original Formulation')
    plt.legend(loc='upper right')
    plt.ylim([0.5, 1])  # Placeholder limits

    plt.show()



" GRAPH 4: Plotting the simulation convergence episodes vs alpha "

# cutoff = 1000  # Define the cutoff index for x values you want to plot

# for index, (sublists, sublists_new) in enumerate(zip(convergence_points_original, convergence_points_1st_shot, convergence_points_2nd_shot)):
#     plt.figure()
#     means, stds = calculate_means_stds(sublists)
#     means_new, stds_new = calculate_means_stds(sublists_new)
    
#     x_values = list(range(len(sublists[0])))
    
#     # Limit to the cutoff index
#     if len(x_values) > cutoff:
#         x_values = x_values[:cutoff]
#         means = means[:cutoff]
#         stds = stds[:cutoff]
#         means_new = means_new[:cutoff]
#         stds_new = stds_new[:cutoff]
        
#     # Plot for original convergence points
#     plt.plot(x_values, means, label=f'Alpha {alpha_values[index]:.2f} Original')
#     plt.fill_between(x_values, np.subtract(means, stds), np.add(means, stds), alpha=0.2)
    
#     # Plot for new convergence points
#     plt.plot(x_values, means_new, label=f'Alpha {alpha_values[index]:.2f} New')
#     plt.fill_between(x_values, np.subtract(means_new, stds_new), np.add(means_new, stds_new), alpha=0.1)
    
#     # Add a horizontal line for the global minimum attribute value
#     plt.axhline(y=min_attribute, color='red', linestyle='dashed', linewidth=2)
#     plt.text(max(x_values)*0.1, min_attribute, f'Global Min: {min_attribute:.4f}', va='top', ha='left', color='red')

#     plt.xlabel('Episode')
#     plt.ylabel('Displacement')
#     plt.title('Comparison of 2-Shot Results vs Original Formulation')
#     plt.legend(loc='upper right')
#     plt.ylim([0.03, 0.1])  # Placeholder limits

#     plt.show()

# " GRAPH 4: Plotting the simulation convergence episodes vs alpha "

# # plt.figure(figsize=(10, 6))  # Optional: Adjust figure size

# for index, sublists in enumerate(convergence_points_NEW_FINAL):
#     # Calculate means and standard deviations for plotting
#     means, stds = calculate_means_stds(sublists)
#     x_values = list(range(len(sublists[0])))

#     # Limit to the cutoff index
#     if len(x_values) > cutoff:
#         x_values = x_values[:cutoff]
#         means = means[:cutoff]
#         stds = stds[:cutoff]
    
    
#     # Plot every simulation run in grey
#     for sublist in sublists:
#         if len(x_values) >= cutoff:
#             sublist = sublist[:cutoff]

#         plt.plot(x_values, sublist, color='grey', alpha=0.5, linewidth=1)

#     # Overlay the mean in blue
#     plt.plot(x_values, means, label=f'alpha value {alpha_values[index]:.2f}', linewidth=2)

#     # Add a horizontal line for the global minimum attribute value
#     plt.axhline(y=min_attribute, color='red', linestyle='dashed', linewidth=2)
#     plt.text(max(x_values), min_attribute, f'Global Minimum: {min_attribute:.4f} (100%)', va='top', ha='right', color='red')
    
    
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.title('Min Result vs Episodes vs. alpha')
#     plt.legend(loc='upper right')
    
#     plt.show()












