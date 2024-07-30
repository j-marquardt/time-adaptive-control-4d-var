import numpy as np

class settings(object):
    """
    Superclass for the "settings_..." subclasses.
    Provides functions which are required in each of the subclasses
    Has to be extended by subclasses which provide the setting for the 
    considered example.
    """
    
    # initialise with default values
    _num_x_elem = None
    _num_t_elem = None
    
    _num_t_plot_elements = None
    
    _t_start = None
    _t_end = None
    
    _x_start = None
    _x_end = None
    
    _alpha = None
    _nu = None
    
    def __str__(self):
        """
        Creates a printable string which contains all important settings
        Is called while printing this class using the print() method
        """
        
        strng  = "####################\n"
        strng += "# Problem settings #\n"
        strng += "####################\n"
        strng += "\n"
        strng += "alpha: " + str(self._alpha) + "\n"
        strng += "nu: " + str(self._nu) + "\n"
        strng += "\n"
        strng += "###########################\n"
        strng += "# Discretization settings #\n"
        strng += "###########################\n"
        strng += "\n"
        strng += "x-space: [" + str(self._x_start) + "," + str(self._x_end) + "]\n"
        strng += "t-space: [" + str(self._t_start) + "," + str(self._t_end) + "]\n"
        strng += "\n"
        strng += "num x-elements: " + str(self.get_num_x_elem()) + "\n"
        strng += "num t-elements: " + str(self.get_num_t_elem()) + "\n"
        strng += "\n"
        strng += "num x nodes: " + str(self.get_num_x_elem() + 1) + "\n"
        strng += "num t nodes: " + str(self.get_num_t_elem() + 1) + "\n"
        strng += "\n"
        
        return strng
    
    # setup functions
    def f(self, t, x):
        """
        Represents the function f in the PDE
        Is not implemented in base class and has to be extended in subclasses
        """
        pass
        
    def y_d(self, t, x):
        """
        Represents the function y_d in the PDE
        Is not implemented in base class and has to be extended in subclasses
        """
        pass
        
    def y_d_t(self, t, x):
        """
        Represents the first derivative in time of the function y_d in the PDE
        Is not implemented in base class and has to be extended in subclasses
        """
        pass
        
    def y_d_Delta(self, t, x):
        """
        Represents the second derivative in space of the function y_d in the PDE
        Is not implemented in base class and has to be extended in subclasses
        """
        pass
        
    def y_b(self, x):
        """
        Represents the function y_b in the PDE
        Is not implemented in base class and has to be extended in subclasses
        """
        pass
        
    
    # get-functions
    
    def get_alpha(self):
        """
        Returns the parameter alpha
        """
        return self._alpha
        
    def get_nu(self):
        """
        Returns the parameter nu
        """
        return self._nu
        
    def get_num_x_elem(self):
        """
        Returns the number of elements in space direction
        """
        return self._num_x_elem
        
    def get_num_t_elem(self):
        """
        Returns the number of elements in time direction
        """
        return self._num_t_elem
        
    def get_num_t_plot_elem(self):
        """
        Returns the number of elements at which the function should be plotted
        (Using only a sparse grid in time while solving the PDE, it might be beneficial
        in some applications to plot the dynamic at more instances)
        """
        if self._num_t_plot_elements is not None:
            return self._num_t_plot_elements
        else:
            return self._num_t_elements
        
    def get_bounds_x(self):
        """
        Returns a tuple with two values: the lower x-bound and upper x-bound
        """
        return self._x_start, self._x_end
        
    def get_bounds_t(self):
        """
        Returns a tuple with two values: the lower t-bound and upper t-bound
        """
        return self._t_start, self._t_end
        