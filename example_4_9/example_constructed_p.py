import sys
sys.path.append("./../src/")

import numpy as np
import matplotlib.pyplot as plt

from helper import *
from settings_constructed_p import *

# save the figures?
save_figures = False

# how many adaptive steps should be executed?
adaptive_steps = 26

# which values of epsilon should be considered?
epsilons = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

####################################
# SETUP AND START OF EPS ITERATION #
####################################

# get the settings
settings = settings_constructed_p(epsilons[0])

# go through all eps
for eps in epsilons:
    
    # make the output in the console look nice
    print((len(str(eps))+ 10) * "#")
    print(f"# eps = {eps} #")
    print((len(str(eps))+ 10) * "#")
    
    settings.set_eps(eps)
    
    # create a grid according to the settings
    grid = helper.grid_from_settings(settings)
    x_space = grid.get_x_space()
    x_num = len(x_space)
    
    # get the true initial condition for comparison later on
    p0_real = np.zeros((x_num,1))
    counter = 0
    for x in x_space:
        p0_real[counter] = settings.p(0,x)
        counter +=1


    # now refine the grid step by step and look how the error between the
    # computed initial condition and true initial condition develops
    
    adaptive_errs = []
    constant_errs = []
    
    for i in range(adaptive_steps):
        num_elem = grid.get_num_t_elements()
        print(f"Adaptive grid t-elements: {num_elem}")
    
        # solve the system on the adaptive grid
        system_matrix, system_rhs = helper.get_fourth_order_system(grid, settings) 
        p = np.linalg.solve(system_matrix, system_rhs)
        p_0 = p[:x_num]
        
        # estimate the error on the adaptive grid
        adaptive_e = sum((p0_real-p_0)**2)[0] / x_num
        print(f"\tAdaptive: {adaptive_e}")
        adaptive_errs.append(adaptive_e)
        
        # create a constant grid with the same number of steps in time
        const_grid = helper.grid_from_settings_const_t_steps(settings, num_elem)
    
        const_system_matrix, const_system_rhs = helper.get_fourth_order_system(const_grid, settings) 
        const_p = np.linalg.solve(const_system_matrix, const_system_rhs)
        const_p_0 = const_p[:len(x_space)]
    
        # estimate the error on the constant grid
        const_e = sum((p0_real-const_p_0)**2)[0] / x_num
        print(f"\tUniform: {const_e}")
        constant_errs.append(const_e)
       
        # grid splitting based on error estimation
        errors = helper.estimate_error(grid, settings)
        arg = np.argmax(errors)
        grid.split_row(arg, 2)

    # create a new figure for each value of eps
    plt.figure()
    plt.semilogy(
        range(settings.get_num_t_elem(), adaptive_steps + settings.get_num_t_elem()),
        adaptive_errs,
        '-',
        label="adaptive")
    plt.semilogy(
        range(settings.get_num_t_elem(), adaptive_steps + settings.get_num_t_elem()),
        constant_errs,
        '--',
        label="uniform")
    
    plt.xlabel('number of timesteps', fontsize=16)
    plt.ylabel('error in $p(0,x)$', fontsize=16)
    plt.legend(fontsize=16)
    
    if save_figures:
        plt.savefig(f"constructed_p_eps/eps={eps}.eps", format="eps", bbox_inches="tight")
    
    # compute the adjoined system state
    system_matrix, system_rhs = helper.get_fourth_order_system(grid, settings) 
    p = np.linalg.solve(system_matrix, system_rhs)
    
    #helper.plot_vec_function(p, grid, settings, label=f"$p(t,x)$ for $\\eps={eps}$")
    grid.plot()
    
    if save_figures:
        plt.savefig(f"constructed_p_grids/eps={eps}.eps", format="eps", bbox_inches="tight")
    
    # make the output look nicer
    print("")
    
plt.show()


            