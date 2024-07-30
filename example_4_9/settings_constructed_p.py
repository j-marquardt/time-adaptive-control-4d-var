from settings import *

class settings_constructed_p(settings):
    
    _num_x_elem = 40
    _num_t_elem = 5
    
    _t_start = 0
    _t_end = 1.0
    
    _x_start = 0
    _x_end = 1
    
    _alpha = 1.0
    _nu = 0.1
    
    _eps = 0.5
    _m = 0.5
    
    def __init__(self, eps):
        self._eps = eps
        
    def set_eps(self, eps):
        self._eps = eps
        
    def g(self, t):
        if t > 0.0:
            return np.exp(-1.0/t)
        else:
            return 0
            
    def g_t(self, t):
        if t > 0.0:
            return np.exp(-1.0/t)/t**2
        else:
            return 0
            
    def g_tt(self, t):
        if t > 0.0:
            return (1-2*t)*np.exp(-1.0/t)/t**4
        else:
            return 0
        
    def p(self, t, x):
    
        frac = self.g(0.5 - (t-self._m)/self._eps)/(self.g(0.5 + (t - self._m)/self._eps) + self.g(0.5 - (t - self._m)/self._eps))
        return frac * np.sin(np.pi*x)
        
    def p_t(self, t, x):
        
        numerator = self.g_t(0.5-(t - self._m)/self._eps)*self.g(0.5+(t - self._m)/self._eps) + self.g(0.5-(t - self._m)/self._eps)*self.g_t(0.5+(t - self._m)/self._eps)
        denominator = self.g(0.5 - (t-self._m)/self._eps) + self.g(0.5 + (t-self._m)/self._eps)
        
        return -1/self._eps * np.sin(np.pi * x) * numerator / denominator**2
        
    def p_tt(self, t, x):
        
        denominator = self.g(0.5+(t - self._m)/self._eps) + self.g(0.5-(t - self._m)/self._eps)
        
        numerator_1 =  (self.g_t(0.5+(t - self._m)/self._eps)-self.g_t(0.5-(t - self._m)/self._eps)) * (self.g(0.5-(t - self._m)/self._eps)*self.g_t(0.5+(t - self._m)/self._eps)+ self.g_t(0.5-(t - self._m)/self._eps)*self.g(0.5+(t - self._m)/self._eps))
        
        numerator_2 = self.g(0.5-(t - self._m)/self._eps)*self.g_tt(0.5+(t - self._m)/self._eps) - self.g_tt(0.5-(t - self._m)/self._eps)*self.g(0.5+(t - self._m)/self._eps)
        
        return np.sin(np.pi*x)/self._eps**2 * (2 * numerator_1 / denominator**3 - numerator_2/denominator**2)
        
    def f(self, t, x):
        
        return -1 * self.p_tt(t,x) + self._nu**2 * np.pi**4 * self.p(t,x) + self.y_d_t(t,x) - self._nu * self.y_d_Delta(t,x)
        
    def y_d(self, t, x):
        return self.p_t(t,x)
        
    def y_d_t(self, t, x):
        
        return self.p_tt(t,x)
        
    def y_d_Delta(self, t, x):
        return - np.pi**2 * self.p_t(t,x)
        
    def y_b(self, x):
        return (1/self._alpha + self._nu * np.pi**2) * self.p(0, x)