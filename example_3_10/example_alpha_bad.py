import sys
sys.path.append("./../src/")

import numpy as np
import matplotlib.pyplot as plt

from helper import *
from settings_alpha_bad import *

# save the figures?
save_figures = False

# values of alpha for which the controls should be considered for
alphas = [0.01, 0.1, 0.25, 0.5, 1.0, 3.0, 10.0]

# the value of alpha for which p should be plotted
plot_p_for = [0.01]


# get settings for an initial choice of alpha
settings = settings_alpha(alphas[0])

# container for controls
controls = []

for alpha in alphas:
    
    ###############################
    # COMPUTE ADJOINT AND CONTROL #
    ###############################
    
    # set settings to current alpha
    settings.set_alpha(alpha)
    
    # create a grid according to the settings
    grid = helper.grid_from_settings(settings)
    
    x_space = grid.get_x_space()
    y_b = settings.y_b(x_space).reshape((len(x_space), 1))

    # compute the adjoined system state
    system_matrix, system_rhs = helper.get_fourth_order_system(grid, settings) 
    p = np.linalg.solve(system_matrix, system_rhs)
    
    if alpha in plot_p_for:
        helper.plot_vec_function(
            p, 
            grid, 
            settings, 
            label=f"$p(t,x)$ for $\\alpha={alpha}$", 
            exp_formatter=True, 
            z_min=-0.017, 
            z_max=0.01)
        
        if save_figures:
            plt.savefig('compare-alpha-bad-p.eps', bbox_inches='tight', pad_inches=0)
    
    # the t=0 nodes are the first nodes in the row
    p_0 = p[:len(x_space)]
    
    # the control can be computed with y_b and p(0)
    control = y_b - 1.0/settings.get_alpha() * p_0
    
    controls.append(control)
    
    #######################################
    # COMPUTE RMSE OF CONTROLLED SOLUTION #
    #######################################
    
    # first of all, we need the dynamics in which the controls result in
    controlled_matrix, controlled_rhs = helper.get_dynamics_system(control, grid, settings)

    # solve the systems
    controlled_dynamics = np.linalg.solve(controlled_matrix, controlled_rhs)
    
    if alpha in plot_p_for:
        helper.plot_vec_function(
            controlled_dynamics, 
            grid, 
            settings, 
            label=f"$y(t,x)$ for $\\alpha={alpha}$", 
            twice_as_long = False)
        
        if save_figures:
            plt.savefig('compare-alpha-bad-y.eps', bbox_inches='tight', pad_inches=0)
        
    # now for the actual computation of the error
    rmse_controlled = 0.0
    
    counter = 0
    for t in grid.get_t_space():
        for x in grid.get_x_space():
            
            rmse_controlled += (settings.y_d(t,x) - controlled_dynamics[counter][0])**2
            counter += 1
            
    rmse_controlled = np.sqrt(1/grid.get_num_nodes() * rmse_controlled)
    
    print(f"alpha = {alpha} => RMSE = {rmse_controlled}")


######################################
# COMPUTE RMSE FROM BACKGROUND GUESS #
######################################

# now also compute the error for the guess as well  
guess_matrix, guess_rhs = helper.get_dynamics_system(y_b, grid, settings)
guess_dynamics = np.linalg.solve(guess_matrix, guess_rhs)

rmse_guess = 0.0

counter = 0
for t in grid.get_t_space():
    for x in grid.get_x_space():
        
        rmse_guess += (settings.y_d(t,x) - guess_dynamics[counter][0])**2
        counter += 1
rmse_guess = np.sqrt(1/grid.get_num_nodes() * rmse_guess)

print(f"From guess => RMSE = {rmse_guess}")

# plot the results
helper.compare_controls(
    controls, 
    settings, 
    grid, 
    labels=["$\\alpha = "+str(alpha) + "$" for alpha in alphas], 
    xvals=[0.25, 0.5] + len(alphas) * [0.75])

if save_figures:
    plt.savefig('compare-alpha-bad-controls.eps', bbox_inches='tight', pad_inches=0)


plt.show()


            