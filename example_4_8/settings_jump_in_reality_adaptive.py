from settings import *

class settings_jump_in_reality_adaptive(settings):
    
    # variables for discretization
    _num_x_elem = 51
    _num_t_elem = 5
    
    _t_start = 0
    _t_end = 1.0
    
    _x_start = 0
    _x_end = 1
    
    # data assimilation / pde parameters
    _alpha = 0.01
    _nu = 0.1
    
    _eps = 0.01
        

    #########################################################
    # functions which occur in the PDE or their derivatives #
    #########################################################
    
    def f(self, t, x):
        return x-x
        
    def y_d(self, t, x):
        return 2.0 * np.sin(np.pi * x) * np.exp(-self._nu*np.pi**2*t) * (1.0/np.pi * np.arctan((t-1/6)/self._eps) + 1.0)
        
    def y_d_t(self, t, x):
        return 2.0 * np.sin(np.pi * x) * np.exp(-self._nu*np.pi**2*t) * (-self._nu * np.pi * np.arctan((t-1/6)/self._eps) - self._nu * np.pi**2 + 1.0/np.pi * self._eps/(self._eps**2 + (t-0.5)**2))
        
    def y_d_Delta(self, t, x):
        return -2.0 * np.pi**2 * np.sin(np.pi * x) * np.exp(-self._nu*np.pi**2*t) * (1.0/np.pi * np.arctan((t-1/6)/self._eps) + 1.0)
        
    def y_b(self, x):
        return np.sin(np.pi * x)