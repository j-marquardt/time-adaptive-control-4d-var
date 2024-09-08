import sys
sys.path.append("./../src/")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np

from helper import *
from settings_show_adjoint import *

# parameters
m = 0.5
eps = 0.5

# save the figures?
save_figures = False

# define the adjoint state and the helper function g
def g(t):
    if t > 0.0:
        return np.exp(-1.0/t)
    else:
        return 0
        
def p(t,x):
    
    frac = g(0.5 - (t-m)/eps)/(g(0.5 + (t-m)/eps) + g(0.5 - (t-m)/eps))
    
    return frac * np.sin(np.pi*x)

######################################
# Start with the 1D plot of p(t,0.5) #
######################################

x_space = np.linspace(0,1,200)
y_space = [p(t,0.5) for t in x_space]

fig, ax = plt.subplots(figsize=(9,9))

# horizontal 0 / 1 lines
plt.plot(x_space, len(x_space) * [0.0], '--', c = "black", linewidth=0.7, dashes=(3, 5))
plt.plot(x_space, len(x_space) * [1.0], '--', c = "black", linewidth=0.7, dashes=(3, 5))

# vertical line center
plt.plot([m, m], [-0.1, 1.1], '--', c = "black", linewidth=0.7, dashes=(3, 5))

# large box
rect_g1 = Rectangle((0.15,-0.1), 0.7, 1.2, facecolor=(1, 0.941, 0.8))
ax.add_patch(rect_g1)


# small box
rect = Rectangle((m-eps/2,-0.1), eps, 1.2, facecolor=(1, 0.847, 0.498))
ax.add_patch(rect)


ax.set_xticks([0.15, 0.5, 0.85], ["$0$", "$m$","$T$"])

# plot all of them
plt.plot(x_space, y_space)

ax.annotate('',
    xy=(m-eps/2, -0.1), 
    xycoords='axes fraction', 
    xytext=(m+eps/2, -0.1), 
    arrowprops=dict(arrowstyle="<->", color=(1, 0.7, 0)))
    
ax.text(m-0.012, -0.3, "$\epsilon$", fontsize=17, color=(1, 0.7, 0))

ax.set_xlabel("$t$", fontsize=16)
ax.xaxis.set_label_coords(1.03, 0.01)

ax.set_ylabel("$p(t,0.5)$", rotation=0, fontsize=16)
ax.yaxis.set_label_coords(0.01, 1.03)
            
plt.gca().set_aspect(1/2)
plt.xlim(0,1)
plt.ylim(-0.1,1.1)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.locator_params(nbins=3)

if save_figures:
    plt.savefig("adjoint_behavior.eps", format="eps", bbox_inches="tight")

#####################
# Surface plot of p #
#####################

# random settings )(for plotting of the grid. Unimportant which functions are implemented how)
settings = settings_show_adjoint()
grid = helper.grid_from_settings(settings)

p_vec = np.zeros((grid.get_num_nodes(),1))

counter = 0
for t in grid.get_t_space():
    for x in grid.get_x_space():
        p_vec[counter] = p(t,x)
        
        counter +=1

helper.plot_vec_function(p_vec, grid, settings, twice_as_long = False)

if save_figures:
    plt.savefig("adjoint_behavior_full.eps", format="eps", bbox_inches="tight")

plt.show()