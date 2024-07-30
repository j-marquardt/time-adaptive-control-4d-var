from settings import *

class settings_alpha(settings):
    
    # variables for discretization
    _num_x_elem = 40
    _num_t_elem = 40
    
    _t_start = 0
    _t_end = 1.0
    
    _x_start = 0
    _x_end = 1
    
    # data assimilation / pde parameters
    _alpha = 1.0
    _nu = 0.1
    
    #####################
    # General functions #
    #####################
    
    def __init__(self, alpha):
        self._alpha = alpha
        
    def set_alpha(self, alpha):
        self._alpha = alpha
        
    
    #########################################################
    # functions which occur in the PDE or their derivatives #
    #########################################################
    
    def f(self, t, x):
        return self.y_d_t(t,x) - self._nu * self.y_d_Delta(t, x)
        
    def y_d(self, t, x):
        
        l = abs(self._x_end - self._x_start)
         
        return np.sin( x * np.pi / l) * np.exp((- np.pi**2 * self._nu * t) / l**2)
        
    def y_d_t(self, t, x):
        l = abs(self._x_end - self._x_start)
        
        return np.sin( np.pi * x / l) * - np.pi**2 * self._nu * np.exp(-np.pi**2 * self._nu * t  / l**2) / l**2
        
    def y_d_Delta(self, t, x):
        l = abs(self._x_end - self._x_start)
        
        return - np.pi**2 * np.sin( np.pi * x / l) * np.exp(-np.pi**2 * self._nu * t / l**2) / l**2
        
    def y_b(self, x):
        return -(x - 0.5)**2 + 0.25