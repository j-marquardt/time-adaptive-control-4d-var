import sys
sys.path.append("./../src/")

import numpy as np
import matplotlib.pyplot as plt

from helper import *
from settings_jump_in_reality import *

# save the figures?
save_figures = False

# get the settings
settings = settings_jump_in_reality()


# create a grid according to the settings
grid = helper.grid_from_settings(settings)

#######################
# PLOT OF THE REALITY #
#######################

# store it in a discretized vector
y_d_vec = np.zeros((grid.get_num_nodes(), 1))

counter = 0
for t in grid.get_t_space():
    for x in grid.get_x_space():
        y_d_vec[counter, 0] = settings.y_d(t,x)
        
        counter += 1

helper.plot_vec_function(
    y_d_vec, 
    grid, 
    settings, 
    twice_as_long = False, 
    label = "y_d", 
    z_max = 2.0)

if save_figures:
    plt.savefig('jump-in-reality-reality.eps', bbox_inches='tight', pad_inches=0)

###########################################
# PLOT THE DYNAMICS FROM BACKGROUND GUESS #
###########################################

x_space = grid.get_x_space()
y_b = settings.y_b(x_space).reshape((len(x_space), 1))

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

print(f"RMSE guess: {rmse_guess}")

helper.plot_vec_function(
    guess_dynamics, 
    grid, 
    settings, 
    twice_as_long = False, 
    label = "background", 
    z_max = 1.1)
    
if save_figures:
    plt.savefig('jump-in-reality-background.eps', bbox_inches='tight', pad_inches=0)


#######################
# PLOT THE DIFFERENCE #
#######################

helper.plot_heatmap_error(
    y_d_vec, 
    guess_dynamics, 
    grid, 
    v_max = 1.6)
    
if save_figures:
    plt.savefig('jump-in-reality-heat.eps', bbox_inches='tight', pad_inches=0)


###################################
# COMPUTE THE CONTROLLED SOLUTION #
###################################

# compute the adjoined system state
system_matrix, system_rhs = helper.get_fourth_order_system(grid, settings) 
p = np.linalg.solve(system_matrix, system_rhs)

helper.plot_vec_function(
    p, 
    grid, 
    settings, 
    label = f"$p(t,x)$", 
    z_max = 0.01, 
    z_min = -0.3)

if save_figures:
    plt.savefig('jump-in-reality-p.eps', bbox_inches='tight', pad_inches=0)

# the t=0 nodes are the first nodes in the row
p_0 = p[:len(x_space)]

# the control can be computed with y_b and p(0)
control = y_b - 1.0/settings.get_alpha() * p_0

helper.compare_controls_same_initial(
    [control], 
    settings, 
    grid, 
    labels = ["$y(0,x)$"], 
    xvals = [0.25, 0.75, 0.25])
    
if save_figures:
    plt.savefig('jump-in-reality-control.eps', bbox_inches='tight', pad_inches=0)

# first of all, we need the dynamics in which the controls result in
controlled_matrix, controlled_rhs = helper.get_dynamics_system(control, grid, settings)

# solve the systems
controlled_dynamics = np.linalg.solve(controlled_matrix, controlled_rhs)

rmse_controlled = 0.0

counter = 0
for t in grid.get_t_space():
    for x in grid.get_x_space():
        
        rmse_controlled += (settings.y_d(t,x) - controlled_dynamics[counter][0])**2
        counter += 1
rmse_controlled = np.sqrt(1/grid.get_num_nodes() * rmse_controlled)

print(f"RMSE controlled: {rmse_controlled}")
    
helper.plot_vec_function(
    controlled_dynamics, 
    grid, 
    settings, 
    label=f"Controlled state $y(t,x)$", 
    twice_as_long = False, 
    z_max = 1.7)
    
if save_figures:
    plt.savefig('jump-in-reality-controlled-state.eps', bbox_inches='tight', pad_inches=0)

helper.plot_heatmap_error(
    y_d_vec, 
    controlled_dynamics, 
    grid, 
    v_max = 1.6)

if save_figures:
    plt.savefig('jump-in-reality-controlled-heat.eps', bbox_inches='tight', pad_inches=0)

plt.show()



            