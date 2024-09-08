from settings import *

class settings_show_adjoint(settings):
    
    # variables for discretization
    _num_x_elem = 40
    _num_t_elem = 40
    
    _t_start = 0
    _t_end = 1.0
    
    _x_start = 0
    _x_end = 1
    
    # data assimilation / pde parameters
    _nu = 0.1

    #########################################################
    # functions which occur in the PDE or their derivatives #
    #########################################################
    
    def f(self, t, x):
        pass
        
    def y_d(self, t, x):
        pass
        
    def y_d_t(self, t, x):
        pass
        
    def y_d_Delta(self, t, x):
        pass
        
    def y_b(self, x):
        pass