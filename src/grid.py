import numpy as np
import matplotlib.pyplot as plt

from element import element

class grid:
    """
    Class which represents the grid on which the PDE can be solved
    Contains the functionality to plot the grid and
    to split an arbitrary time-row
    """
    
    # initialise with default values
    _t_min = None
    _t_max = None
	
    _x_min = None
    _x_max = None
    
    _num_x_elem = 0
    _num_t_elem = 0
    
    _space_time_elements = None
    
    _start_node_numbers  = None
    _T_node_numbers = None
    _left_boundary_node_numbers = None
    _right_boundary_node_numbers = None
    
    _x_space = None
    _t_space = None

    def __init__(self, x_min, x_max, t_min, t_max, num_x_elem, num_t_elem):
        """
        Stores the data and calls the create elements method in order to create a
        representation of a grid.
        """
        
        self._x_min = x_min
        self._x_max = x_max
        self._t_min = t_min
        self._t_max = t_max
        
        self._num_x_elem = num_x_elem
        self._num_t_elem = num_t_elem
        
        self._x_space = np.linspace(self._x_min, self._x_max, self._num_x_elem + 1, endpoint = True)
        self._t_space = np.linspace(self._t_min, self._t_max, self._num_t_elem + 1, endpoint = True)
        
        self.create_elements()
        
    def __str__(self):
        """
        Returns a string wich represents this element
        Is called while printing this class using the print() method
        """
        
        strng = "##### Grid settings #####\n"
        strng += "\n"
        strng += "x-space: [" + str(self._x_min) + "," + str(self._x_max) + "]\n"
        strng += "t-space: [" + str(self._t_min) + "," + str(self._t_max) + "]\n"
        strng += "\n"
        strng += "num x-elements: " + str(self._num_x_elem) + "\n"
        strng += "num t-elements: " + str(self._num_t_elem) + "\n"
        strng += "\n"
        strng += "num x nodes: " + str(self._num_x_elem + 1) + "\n"
        strng += "num t nodes: " + str(self._num_t_elem + 1) + "\n"
        strng += "\n"
        
        return strng
        
    def create_elements(self):
        """
        Elements are created in this method.
        The method will be called upon creation of the grid.
        It can be called whenever settings have changed and
        a full recomputation of the elements is required.
        
        The elements are numerated as follows:
        7---8---9 t2
        |   |   |
        4---5---6 t1
        |   |   |
        1---2---3 t0
        x0  x1  x2
        """
        
        self._space_time_elements = []
        self. _start_node_numbers  = []
        self._T_node_numbers = []
        self._left_boundary_node_numbers = []
        self._right_boundary_node_numbers = []
        
        for t_counter, t in enumerate(self._t_space[:-1]):
            
            dt = self._t_space[t_counter + 1] - t
            
            for x_counter, x in enumerate(self._x_space[:-1]):
                
                dx = self._x_space[x_counter + 1] - x
                
                # initialise element with euclidean coordinates in the grid
                e = element(x, x + dx, t, t + dt)
                
                # add the grid-point coordinates
                
                bottom_left = t_counter * self.get_num_x_nodes() + x_counter
                bottom_right = t_counter * self.get_num_x_nodes() + x_counter + 1
                top_left = (t_counter + 1) * self.get_num_x_nodes() + x_counter
                top_right = (t_counter + 1) * self.get_num_x_nodes() + x_counter + 1
                
                e.set_corner_numbers(
                                    bottom_left,
                                    bottom_right,
                                    top_right,
                                    top_left)
                
                e.set_t_row_number(t_counter)
                
                # set marker if the element is in the first row
                if t_counter == 0:
                    e.indicate_first_row()
                    
                    if x_counter == 0:
                        self._start_node_numbers.append(bottom_left)
                    
                    self._start_node_numbers.append(bottom_right)
                    
                # set marker if the element is in the last row
                if t_counter == self._num_t_elem - 1:
                    e.indicate_last_row()
                    
                    if x_counter == 0:
                        self._T_node_numbers.append(top_left)
                    self._T_node_numbers.append(top_right)
                    
                # set marker if the element is a left-x boundary node
                if x_counter == 0:
                    e.indicate_left_boundary()
                    
                    if t_counter == 0:
                        self._left_boundary_node_numbers.append(bottom_left)
                    
                    self._left_boundary_node_numbers.append(top_left)
                    
                # set marker if the element is a right-x boundary node
                if x_counter == self._num_x_elem - 1:
                    e.indicate_right_boundary()
                    
                    if t_counter == 0:
                        self._right_boundary_node_numbers.append(bottom_right)
                    
                    self._right_boundary_node_numbers.append(top_right)
                
                # append the element to the grid
                self._space_time_elements.append(e)
        
    def plot(self, show_node_names = False):
        """
        Prepares the plotting of the grid.
        Does not call the plt.show() function.
        """
        
        plt.figure()
        
        # go through all elements
        for e in self._space_time_elements:
            
            t_min, t_max = e.get_t_bounds()
            x_min, x_max = e.get_x_bounds()
            
            # plot only right and top boundary of each element on default
            # such that lines do not overlay when plotting multiple elements
            plt.plot([x_max, x_max], [t_min, t_max], c="b", linewidth='0.3')
            plt.plot([x_min, x_max], [t_max, t_max], c="r", linewidth='0.4')
            
            # if the element is t0 or x0 element, plot in addition the left and lower boundary.
            if e.is_t0_element():
                plt.plot([x_min, x_max], [t_min, t_min], c="r", linewidth='0.4')
                
            if e.is_x_min_element():
                plt.plot([x_min, x_min], [t_min, t_max], c="b", linewidth='0.3')
              
            # if required, also plot the node names at the nodes 
            if show_node_names:
                
                c1, c2, c3, c4 = e.get_corner_numbers()
                
                coordinates = [c1]
                
                if e.is_x_max_element():
                    coordinates.append(c2)
                    
                if e.is_tend_element():
                    coordinates.append(c4)
                    
                if e.is_x_max_element() and e.is_tend_element():
                    coordinates.append(c3)
                
                for c in coordinates:
                    t, x = self.get_tx_by_node_num(c)
                    plt.text(x, t, str(c))
                    
        plt.xlabel('$x$', fontsize=16)
        plt.ylabel('$t$', fontsize=16)
            
    
    def split_row(self, row_no, into):
        """
        Splits the given row into 'into' number of rows.
        """
         
        self._num_t_elem += into - 1
        
        row_start_t = self._t_space[row_no]
        row_end_t = self._t_space[row_no + 1]
        new_dt =  (row_end_t - row_start_t) / into
        
        for i in range(into - 1):
            self._t_space = np.insert(self._t_space, row_no + i + 1, row_start_t + (i + 1) * new_dt)
        
        self.create_elements()
        
                
            
        
    def get_tx_by_node_num(self, num):
        """
        Returns the (t,x) coordinates of a node by its number
        """
        
        row_no = num // (self._num_x_elem + 1)
        col_no = num % (self._num_x_elem + 1)
        
        
        return self._t_space[row_no], self._x_space[col_no]
        
    def get_elements(self):
        """
        Returns all the elements of the grid. Notice that they may not be ordered
        """
        return self._space_time_elements
        
    def get_x_space(self):
        """
        Returns a numpy array which represents the nodes in x-direction
        """
        return self._x_space
        
    def get_t_space(self):
        """
        Returns a numpy array which represents the nodes in t-direction
        """
        return self._t_space
            
    def get_num_x_nodes(self):
        """
        Returns the number of x nodes in the grid.
        """
        return self._num_x_elem + 1
            
    def get_num_t_nodes(self):
        """
        Returns the number of t nodes in the grid.
        """
        return self._num_t_elem + 1
        
    def get_num_t_elements(self):
        """
        Returns the number of t elements in the grid.
        """
        return self._num_t_elem
        
    def get_num_nodes(self):
        """
        Returns the total number nodes in the grid.
        """
        return (self._num_x_elem + 1) * (self._num_t_elem + 1)
        
    def get_boundary_node_numbers(self):
        """
        Returns the number of boundary node numbers on BOTH the left and right boundary combined.
        """
        return self._left_boundary_node_numbers + self._right_boundary_node_numbers
        
    def get_T_node_numbers(self):
        """
        Returns the node numbers of the nodes, which are located at the end if time where t=T
        """
        return self._T_node_numbers
        
    def get_0_node_numbers(self):
        """
        Returns the node numbers of the nodes, which are located at the beginning if time where t=0
        """
        return self._start_node_numbers
        
        