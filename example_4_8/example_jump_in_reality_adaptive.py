import sys
sys.path.append("./../src/")

import numpy as np
import matplotlib.pyplot as plt

from helper import *
from settings_jump_in_reality_adaptive import *


# save the figures?
save_figures = False

# how many adaptive steps should be executed?
adaptive_steps = 35

# plot p0 after how many adaptive steps?
output_initial_iters = [5, 10, 20]

####################################################
# SETUP AND COMPUTATION OF REALITY (HIFI SOLUTION) #
####################################################

# containers to store the error values in
error_estimates = []
error_constant = []
error_true = []

# get the settings
settings = settings_jump_in_reality_adaptive()

# create a grid according to the settings
grid = helper.grid_from_settings(settings)

x_space = grid.get_x_space()
x_num = len(x_space)

# compute a hifi solution which we can assume to be the reality
# (or at least closer to the reality)
hifi_grid = helper.grid_from_settings_const_t_steps(settings)

hifi_system_matrix, hifi_system_rhs = helper.get_fourth_order_system(hifi_grid, settings)
hifi_p = np.linalg.solve(hifi_system_matrix, hifi_system_rhs)
hifi_p_0 = hifi_p[:x_num]

# execute the adaptive steps 
for i in range(adaptive_steps):
    
    ###################
    # GRID REFINEMENT #
    ###################
    
    num_elem = grid.get_num_t_elements()
    print(f"Adaptive grid t-elements: {num_elem}")
    
    errors = helper.estimate_error(grid, settings)
    arg = np.argmax(errors)
    grid.split_row(arg, 2)
    
    # store error estimate for comparison
    error_estimates.append(sum(errors))
    
    ###########################################
    # COMPARE P ON CONSTANT AND ADAPTIVE GRID #
    ###########################################
    
    # create a constant grid with the same number of steps in time
    const_grid = helper.grid_from_settings_const_t_steps(settings, num_elem)

    const_system_matrix, const_system_rhs = helper.get_fourth_order_system(const_grid, settings) 
    const_p = np.linalg.solve(const_system_matrix, const_system_rhs)
    const_p_0 = const_p[:len(x_space)]
    
    # store error in initial condition
    error_constant.append(np.linalg.norm(hifi_p_0 - const_p_0))
    
    # the same now on the adaptive grid
    system_matrix, system_rhs = helper.get_fourth_order_system(grid, settings) 
    p = np.linalg.solve(system_matrix, system_rhs)
    p_0 = p[:x_num]
    
    # plot p_0 if required
    if i in output_initial_iters:
        
        plt.figure()
        
        plt.plot(hifi_grid.get_x_space(), hifi_p_0, label="hifi")
        plt.plot(grid.get_x_space(), p_0, '-.', label="adaptive")
        plt.plot(const_grid.get_x_space(), const_p_0, '--', label="uniform")
        
        plt.xlabel('$x$', fontsize=16)
        plt.legend(fontsize=16)
        plt.locator_params(nbins=3)
        
        if save_figures:
            plt.savefig(f"jump_adaptive_p0_{i}.eps", format="eps", bbox_inches="tight")
    
    # store error in initial condition
    error_true.append(np.linalg.norm(hifi_p_0 - p_0))

#########################################
# CREATE GRAPHICS OF THE COMPLETED GRID #
#########################################
    
# plot p on the adaptive grid
helper.plot_vec_function(
    p, 
    grid, 
    settings, 
    label = f"$p(t,x)$", 
    z_min = -0.3, 
    z_max = 0.1)
    
if save_figures:
    plt.savefig("jump_adaptive_p.eps", format="eps", bbox_inches="tight")
    
# plot the errors collected during the iterations
plt.figure()
plt.semilogy(
    range(settings.get_num_t_elem(), adaptive_steps + settings.get_num_t_elem()), 
    np.sqrt(error_estimates),
    label="error estimate")

plt.semilogy(
    range(settings.get_num_t_elem(), adaptive_steps + settings.get_num_t_elem()), 
    error_true, 
    '-.', 
    label="adaptive")
    
plt.semilogy(
    range(settings.get_num_t_elem(), adaptive_steps + settings.get_num_t_elem()), 
    error_constant,
    '--',
    label="uniform")

plt.xlabel('number of timesteps', fontsize=16)
plt.ylabel('error in $p(0,x)$ and error estimate', fontsize=16)
plt.legend(fontsize=16)

if save_figures:
    plt.savefig("jump_adaptive_error.eps", format="eps", bbox_inches="tight")

# plot the grid
grid.plot()

if save_figures:
    plt.savefig("jump_adaptive_grid.eps", format="eps", bbox_inches="tight")
    
plt.show()