class element:
    """
    Represents an element in the grid and stores all important information about the element,
    i.e. the position of its edges.
    Also contains helper functions for the implementations of finite elements
    """
    
    # variable initialisation
    _t_min = None
    _t_max = None
	
    _x_min = None
    _x_max = None
    
    _c1_num_in_grid = -1
    _c2_num_in_grid = -1
    _c3_num_in_grid = -1
    _c4_num_in_grid = -1
    
    _t0_element = False
    _tend_element = False
    
    _left_boundary = False
    _right_boundary = False
    
    _t_row_number = None

    def __init__(self, x_min, x_max, t_min, t_max):
        """
        Initialieses the element by setting the t and x min/max values of the element
        """
        self._t_min = t_min
        self._t_max = t_max
        self._x_min = x_min
        self._x_max = x_max
        
        
    def __str__(self):
        """
        Returns a string wich represents this element and contains the position of the edges.
        Is called while printing this class using the print() method
        """
        return f"({self._x_min},{self._t_min}) - ({self._x_max},{self._t_max})"
     
    # set-fucntions  
    def set_corner_numbers(self, c1, c2, c3, c4):
        """
        Assign each corner of the element its number in the grid
        """
        self._c1_num_in_grid = c1
        self._c2_num_in_grid = c2
        self._c3_num_in_grid = c3
        self._c4_num_in_grid = c4
        
    def set_corner_values(self, x_min, x_max, t_min, t_max):
        """
        Set the t and x min/max values of the element
        """
        self._t_min = t_min
        self._t_max = t_max
        self._x_min = x_min
        self._x_max = x_max
        
    def set_t_row_number(self, number):
        """
        Define in which time-row in the grid this element is located
        """
        self._t_row_number = number
        
    def indicate_first_row(self):
        """
        Define whether this is an element in the first time row
        """
        self._t0_element = True
        
    def indicate_last_row(self):
        """
        Define whether this is an element in the last time row
        """
        self._tend_element = True
        
    def remove_last_row_indication(self):
        """
        Indicate, that this element is not in the last row anymore 
        (Required i.e. during the splitting of rows in t-direction)
        """
        self._tend_element = False
        
    def indicate_left_boundary(self):
        """
        Define whether this is an element in the first space column
        """
        self._left_boundary = True
        
    def indicate_right_boundary(self):
        """
        Define whether this is an element in the last space column
        """
        self._right_boundary = True
    
    # get-functions
    def is_t0_element(self):
        """
        Returns true, if element is in the first time row, i.e. if it describes the system for t=0
        """
        return self._t0_element
        
    def is_tend_element(self):
        """
        Returns true, if element is in the last time row, i.e. if it describes the system for t=T
        """
        return self._tend_element
        
    def is_x_min_element(self):
        """
        Returns true, if element is adjacent to the left boundary
        """
        return self._left_boundary
        
    def is_x_max_element(self):
        """
        Returns true, if element is adjacent to the right boundary
        """
        return self._right_boundary
        
    def get_x_bounds(self):
        """
        Returns the x-boundary values of this element.
        """
        return self._x_min, self._x_max
        
    def get_t_bounds(self):
        """
        Returns the t-boundary values of this element.
        """
        return self._t_min, self._t_max
        
    def get_corner_numbers(self):
        """
        Returns the number of the corners in the grid as a list.
        The order is
        4---3
        |   |
        1---2
        """
        
        return [self._c1_num_in_grid, self._c2_num_in_grid, self._c3_num_in_grid, self._c4_num_in_grid]
        
    def get_t_row_number(self):
        """
        Returns the number of the time-row in which the element is located
        (starting with 0)
        """
        return self._t_row_number
    
    def get_trans_fac(self):
        """
        Returns the factor |det(D_phi)| needed in the transformation theorem,
        while switching from this element to the reference element [0,1]x[0,1]
        """
        
        return self.get_dt() * self.get_dx()
        
    def get_phi_inv_x(self):
        """
        Returns the x-component of the derivative in x direction of the inverse function
        of the transformation mapping phi.
        Phi is the mapping which maps the element onto the reference element [0,1]x[0,1]
        """
        
        # Phi = b + Ax, where A is diagonal with dt and dx on its diagonal
        # => x-component of x-derivative is dx
        
        return 1.0/self.get_dx()
        
    def get_phi_inv_t(self):
        """
        Returns the t-component of the derivative in t direction of the inverse function
        of the transformation mapping phi.
        Phi is the mapping which maps the element onto the reference element [0,1]x[0,1]
        """
        
        # Phi = b + Ax, where A is diagonal with dt and dx on its diagonal
        # => t-component of t-derivative is dt
        
        return 1.0/self.get_dt()
        
    def get_dx(self):
        """
        Returns the diameter in x-direction of this element
        """
        return abs(self._x_max - self._x_min)
        
    def get_dt(self):
        """
        Returns the diameter in t-direction of this element
        """
        return abs(self._t_max - self._t_min)