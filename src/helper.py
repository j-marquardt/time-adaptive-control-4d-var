import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm
from labellines import labelLines

from matplotlib.ticker import LinearLocator
import sys

import numpy as np
from scipy import integrate

from grid import grid

class helper:
    """
    Class which contains only static functions and variables, 
    which can perform the tasks required for i.e. the fe discretization,
    error estimation or plotting
    """
    
    #################################################
    # reference matrices which are used to compute  #
    # the mass and stiffness matrices of the system #
    #################################################
    
    # mass matrix for space-time integration
    mass_ref_matrix = np.array([
        [1/9, 1/18, 1/36, 1/18],
        [1/18, 1/9, 1/18, 1/36],
        [1/36, 1/18, 1/9, 1/18],
        [1/18, 1/36, 1/18, 1/9]
    ])
    
    # stiffness matrix for space-time integration with respect to space derivatives
    stiff_x_ref_matrix = np.array([
        [1/3, -1/3, -1/6, 1/6],
        [-1/3, 1/3, 1/6, -1/6],
        [-1/6, 1/6, 1/3, -1/3],
        [1/6, -1/6, -1/3, 1/3]
    ])
    
    # stiffness matrix for space-time integration with respect to time derivatives
    stiff_t_ref_matrix = np.array([
        [1/3, 1/6, -1/6, -1/3],
        [1/6, 1/3, -1/3, -1/6],
        [-1/6, -1/3, 1/3, 1/6],
        [-1/3, -1/6, 1/6, 1/3]
    ])
    
    # stiffness matrix for space-time integration with respect to only one time derivative
    stiff_semi_t_ref_matrix = np.array([
        [-1/6, -1/12, 1/12, 1/6],
        [-1/12, -1/6, 1/6, 1/12],
        [-1/12, -1/6, 1/6, 1/12],
        [-1/6, -1/12, 1/12, 1/6]
    ])
    
    # mass matrix for integration at t=0
    mass_0_ref_matrix = np.array([
        [1/3, 1/6],
        [1/6, 1/3]
    ])
    
    # stiffness matrix for integration at t=0
    stiff_0_ref_matrix = np.array([
        [1,-1],
        [-1, 1]
    ])
    
    ###################################################
    # everything needed for gauss quadrature with n=4 #
    ###################################################
    
    # gauss weights for numerical integration
    gauss_weights = [
        0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454
    ]
    
    # corresponding nodes for numerical integration
    gauss_nodes = [
        -0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053
    ]
    
    ################################################
    # methods which help to create and plot a grid #
    ################################################
    
    
    def grid_from_settings(settings):
        """
        Creates a grid which fits the params given with the settings
        """
        
        x_min, x_max = settings.get_bounds_x()
        t_min, t_max = settings.get_bounds_t()
        
        num_x_elem = settings.get_num_x_elem()
        num_t_elem = settings.get_num_t_elem()
        
        return grid(x_min, x_max, t_min, t_max, num_x_elem, num_t_elem)
        
    def grid_from_settings_const_t_steps(settings, steps = 0):
        """
        Creates a grid which fits the params given with the settings
        """
        
        if steps == 0:
            steps = settings.get_num_x_elem()
        
        x_min, x_max = settings.get_bounds_x()
        t_min, t_max = settings.get_bounds_t()
        
        num_x_elem = settings.get_num_x_elem()
        
        return grid(x_min, x_max, t_min, t_max, num_x_elem, steps)
        
    def plot_grid_from_settings(settings):
        """
        Creates a grid which fits the params given with the settings
        with the differing number of plot elements from the settings.
        Requires the setting to define the number of plot elements
        """
        
        x_min, x_max = settings.get_bounds_x()
        t_min, t_max = settings.get_bounds_t()
        
        num_x_elem = settings.get_num_x_elem()
        num_t_elem = settings.get_num_t_plot_elem()
        
        return grid(x_min, x_max, t_min, t_max, num_x_elem, num_t_elem)
    
        
    #################################################
    # methods fot the finite element discretization #
    #################################################
    
    def ref_basis_func(k, t, x):
        """
        The basis function on the reference element [0,1]x[0,1].
        Numbering for k is 
        4---3
        |   |
        1---2
        """
        if k == 0:
            return (1.0 - t) * (1.0 - x)
        elif k == 1:
            return (1.0 - t) * x
        elif k == 2:
            return t * x
        elif k ==3:
            return t * (1.0 - x)
        else:
            sys.exit("ERROR: Reference basis function can only be returned for k <= 3")
    
    def get_fourth_order_y_dynamics(control, grid, settings):
        """
        Returns the system matrix and the system rhs of the fourth order pde in y
        All initial and endtime conditions and boundary conditions are already incorporated 
        """
        
        # total number of grid points
        nodes_num = grid.get_num_nodes()
        
        # allocate space for matrices
        mass_matrix = np.zeros((nodes_num, nodes_num))
        stiff_matrix_x = np.zeros((nodes_num, nodes_num))
        stiff_matrix_t = np.zeros((nodes_num, nodes_num))
        
        stiff_matrix_x_T = np.zeros((nodes_num, nodes_num))
        
        # allocate space for right hand side
        rhs = np.zeros((nodes_num, 1))
        rhs_T = np.zeros((nodes_num, 1))
        rhs_boundary = np.zeros((nodes_num, 1))

        # get the integrals for the reference element [0,1]x[0,1]
        mass_ref_matrix = helper.mass_ref_matrix
        stiff_x_ref_matrix = helper.stiff_x_ref_matrix
        stiff_t_ref_matrix = helper.stiff_t_ref_matrix

        stiff_T_ref_matrix = helper.stiff_0_ref_matrix
        
        for e in grid.get_elements():
            
            # number of the corners in the grid as list
            coordinates = e.get_corner_numbers()
    
            # factor for transformation theorem 
            trans_fac = e.get_trans_fac()
    
            # get inverse of transformation function for transformation theorem later on
            phi_inv_x = e.get_phi_inv_x()
            phi_inv_t = e.get_phi_inv_t()
    
            # precomputation for stiffness matrices
            sx = trans_fac * phi_inv_x * phi_inv_x 
            st = trans_fac * phi_inv_t * phi_inv_t 
    
            sT = e.get_dx() * phi_inv_x * phi_inv_x
            
            for count_i, i in enumerate(coordinates):
                
                # compute rhs once for ever row
                rhs[i, 0] = rhs[i, 0] + helper.rhs_y_integral(count_i, e, settings)
        
                # rhs part which is the integral over Omega for t=T
                if e.is_tend_element() and count_i >= 2:
                    rhs_T[i, 0] = rhs_T[i, 0] + helper.rhs_y_integral_T(count_i, e, settings)
                    
                for count_j, j in enumerate(coordinates):
            
                    # mass matrix with transformation theorem
                    mass_matrix[i,j] = mass_matrix[i,j] + trans_fac * mass_ref_matrix[count_i,count_j]
            
                    # stiffness matrices require more than just multiplication with det in transformation theorem
                    # the derivatives require the application of the chain rule
                    stiff_matrix_x[i,j] = stiff_matrix_x[i,j] + sx * stiff_x_ref_matrix[count_i,count_j]
                    stiff_matrix_t[i,j] = stiff_matrix_t[i,j] + st * stiff_t_ref_matrix[count_i,count_j]
            
                    # compute all integrals occurring for t=0
                    if e.is_tend_element() and count_i >= 2 and count_j >= 2:
                        # mass and stiffness matrices
                        stiff_matrix_x_T[i,j] = stiff_matrix_x_T[i,j] + sT * stiff_T_ref_matrix[count_i-2,count_j-2]
        
        
        # assemble system matrices and everything else
        system_matrix_11 = stiff_matrix_t + settings.get_nu() * stiff_matrix_x_T
        system_matrix_12 = settings.get_nu() * stiff_matrix_x
        system_matrix_21 = - settings.get_nu() * stiff_matrix_x
        system_matrix_22 = mass_matrix
        
        ###############################################
        # incorporate control and boundary conditions #
        ###############################################

        ### y(0) = u in Omega
        start_time_nodes = grid.get_0_node_numbers()
        
        for k in start_time_nodes:
    
            # start with right hand side:
            rhs[k, 0] = control[k, 0]
    
            # manipulate the k_th row which is responsible for the k-th start value
            for j in range(nodes_num):
                system_matrix_11[k, j] = 1 if k == j else 0
                system_matrix_12[k, j] = 0
        
        
        
        ### y = 0 on spacial boundary
        ### z = f on spacial boundary
        
        x_boundary_nodes = grid.get_boundary_node_numbers()

        for k in x_boundary_nodes:
    
            # right hand side for y=0 property:
            rhs[k, 0] = 0
            rhs_T[k, 0] = 0
    
            # right hand side for z = f property:
            t,x = grid.get_tx_by_node_num(k)
            rhs_boundary[k, 0] = - settings.f(t,x)
    
            # manipulate the k_th row which is responsible for the k-th boundary value
            for j in range(nodes_num):
        
                # matrices for y = 0 property
                system_matrix_11[k, j] = 1 if k == j else 0
                system_matrix_12[k, j] = 0
        
                # matrices 
                system_matrix_22[k, j] = 1 if k == j else 0
                system_matrix_21[k, j] = 0
        
        system_matrix = np.block([
            [system_matrix_11, system_matrix_12],
            [system_matrix_21, system_matrix_22]
        ])
    
    
        system_rhs = np.block([
            [rhs + rhs_T],
            [rhs_boundary]
        ])
        
        return system_matrix, system_rhs
      
    def get_fourth_order_system(grid, settings):
        """
        Returns the system matrix and the system rhs of the fourth order pde in p
        All initial and endtime conditions and boundary conditions are already incorporated 
        """
        
        # total number of grid points
        nodes_num = grid.get_num_nodes()
        
        # allocate space for matrices
        mass_matrix = np.zeros((nodes_num, nodes_num))
        stiff_matrix_x = np.zeros((nodes_num, nodes_num))
        stiff_matrix_t = np.zeros((nodes_num, nodes_num))
        
        mass_matrix_0 = np.zeros((nodes_num, nodes_num))
        stiff_matrix_x_0 = np.zeros((nodes_num, nodes_num))
        
        # allocate space for right hand side
        rhs = np.zeros((nodes_num, 1))
        rhs_0 = np.zeros((nodes_num, 1))
        rhs_boundary = np.zeros((nodes_num, 1))

        # get the integrals for the reference element [0,1]x[0,1]
        mass_ref_matrix = helper.mass_ref_matrix
        stiff_x_ref_matrix = helper.stiff_x_ref_matrix
        stiff_t_ref_matrix = helper.stiff_t_ref_matrix

        mass_0_ref_matrix = helper.mass_0_ref_matrix
        stiff_0_ref_matrix = helper.stiff_0_ref_matrix
        
        
        for e in grid.get_elements():
    
            # number of the corners in the grid as list
            coordinates = e.get_corner_numbers()
    
            # factor for transformation theorem 
            trans_fac = e.get_trans_fac()
    
            # get inverse of transformation function for transformation theorem later on
            phi_inv_x = e.get_phi_inv_x()
            phi_inv_t = e.get_phi_inv_t()
    
            # precomputation for stiffness matrices
            sx = trans_fac * phi_inv_x * phi_inv_x 
            st = trans_fac * phi_inv_t * phi_inv_t 
    
            s0 = e.get_dx() * phi_inv_x * phi_inv_x
    
            for count_i, i in enumerate(coordinates):
        
                # compute rhs once for ever row
                rhs[i, 0] = rhs[i, 0] + helper.rhs_integral(count_i, e, settings)
        
                # rhs part which is the integral over Omega for t=0
                if e.is_t0_element() and count_i <= 1:
                    rhs_0[i, 0] = rhs_0[i, 0] + helper.rhs_integral_0(count_i, e, settings)
            
                for count_j, j in enumerate(coordinates):
            
                    # mass matrix with transformation theorem
                    mass_matrix[i,j] = mass_matrix[i,j] + trans_fac * mass_ref_matrix[count_i,count_j]
            
                    # stiffness matrices require more than just multiplication with det in transformation theorem
                    # the derivatives require the application of the chain rule
                    stiff_matrix_x[i,j] = stiff_matrix_x[i,j] + sx * stiff_x_ref_matrix[count_i,count_j]
                    stiff_matrix_t[i,j] = stiff_matrix_t[i,j] + st * stiff_t_ref_matrix[count_i,count_j]
            
                    # compute all integrals occurring for t=0
                    if e.is_t0_element() and count_i <= 1 and count_j <= 1:
                        # mass and stiffness matrices
                        mass_matrix_0[i,j] = mass_matrix_0[i,j] + e.get_dx() * mass_0_ref_matrix[count_i,count_j] # hier ggf noch mit dt multiplizieren??
                        stiff_matrix_x_0[i,j] = stiff_matrix_x_0[i,j] + s0 * stiff_0_ref_matrix[count_i,count_j]
                
                
        
        
        # assemble system matrices and everything else
        system_matrix_11 = stiff_matrix_t + 1.0 / settings.get_alpha() * mass_matrix_0 + settings.get_nu() * stiff_matrix_x_0
        system_matrix_12 = settings.get_nu() * stiff_matrix_x
        system_matrix_21 = - settings.get_nu() * stiff_matrix_x
        system_matrix_22 = mass_matrix
        
        ################################################
        # incorporate end-time and boundary conditions #
        ################################################

        ### p(T) = 0 in Omega

        end_time_nodes = grid.get_T_node_numbers()
        

        for k in end_time_nodes:
    
            # start with right hand side:
            rhs[k, 0] = 0
    
            # manipulate the k_th row which is responsible for the k-th end value
            for j in range(nodes_num):
                system_matrix_11[k, j] = 1 if k == j else 0
                system_matrix_12[k, j] = 0

        ### p = 0 on spacial boundary
        ### -q = y_d on spacial boundary

        x_boundary_nodes = grid.get_boundary_node_numbers()

        for k in x_boundary_nodes:
    
            # right hand side for p=0 property:
            rhs[k, 0] = 0
            rhs_0[k, 0] = 0
    
            # right hand side for -q = y_d property:
            t,x = grid.get_tx_by_node_num(k)
            rhs_boundary[k, 0] = settings.y_d(t,x)
    
            # manipulate the k_th row which is responsible for the k-th boundary value
            for j in range(nodes_num):
        
                # matrices for p = 0 property
                system_matrix_11[k, j] = 1 if k == j else 0
                system_matrix_12[k, j] = 0
        
                # matrices 
                system_matrix_22[k, j] = 1 if k == j else 0
                system_matrix_21[k, j] = 0
    

    
        system_matrix = np.block([
            [system_matrix_11, system_matrix_12],
            [system_matrix_21, system_matrix_22]
        ])
    
    
        system_rhs = np.block([
            [rhs + rhs_0],
            [rhs_boundary]
        ])
        
        return system_matrix, system_rhs
        
        
    def get_dynamics_system(initial_cond, grid, settings):
        """
        returns a system matrix and a right hand side of the simulated parabolic equation
        for a given initial condition.
        Space-time finite elements are used for that.
        Boundary values are already included, initial values are passed to this function.
        """
        
        if initial_cond.shape[0] != grid.get_num_x_nodes():
            sys.exit("ERROR: Shape of initial condition does not match nodes number!")
            
        # total number of grid points
        nodes_num = grid.get_num_nodes()
        
        # allocate space for matrices
        stiff_matrix_x = np.zeros((nodes_num, nodes_num))
        stiff_matrix_semi_t = np.zeros((nodes_num, nodes_num))
        
        # allocate space for right hand side
        rhs = np.zeros((nodes_num, 1))

        # get the integrals for the reference element [0,1]x[0,1]
        stiff_x_ref_matrix = helper.stiff_x_ref_matrix
        stiff_semi_t_ref_matrix = helper.stiff_semi_t_ref_matrix

        for e in grid.get_elements():
            
    
            # number of the corners in the grid as list
            coordinates = e.get_corner_numbers()
            
            #print(f"Element: {coordinates}")
    
            # factor for transformation theorem 
            trans_fac = e.get_trans_fac()
            
            # get inverse of transformation function for transformation theorem later on
            phi_inv_x = e.get_phi_inv_x()
            phi_inv_t = e.get_phi_inv_t()
    
            # precomputation for stiffness matrices
            sx = trans_fac * phi_inv_x * phi_inv_x 
            st = trans_fac * phi_inv_t
    
            for count_i, i in enumerate(coordinates):
        
                # compute rhs once for ever row
                rhs[i, 0] = rhs[i, 0] + helper.rhs_dynamics_integral(count_i, e, settings)
            
                for count_j, j in enumerate(coordinates):
                    
                    stiff_matrix_x[i,j] += sx * stiff_x_ref_matrix[count_i,count_j]
                    stiff_matrix_semi_t[i,j] += st * stiff_semi_t_ref_matrix[count_i,count_j]
        
        system_matrix = stiff_matrix_semi_t + settings.get_nu() * stiff_matrix_x
        

        ####################################################
        # incorporate initial-time and boundary conditions #
        ####################################################
        
        ### y(0) = initial cond in Omega

        start_time_nodes = grid.get_0_node_numbers()
        #print(f"0-nodes: {start_time_nodes}")

        for k in start_time_nodes:
    
            # start with right hand side:
            rhs[k, 0] = initial_cond[k, 0]
    
            # manipulate the k_th row which is responsible for the k-th end value
            for j in range(nodes_num):
                system_matrix[k, j] = 1 if k == j else 0
                
        ### y = 0 on spacial boundary
        
        x_boundary_nodes = grid.get_boundary_node_numbers()
        
        #print(f"Boundary nodes: {x_boundary_nodes}\n")
        
        for k in x_boundary_nodes:
    
            # right hand side for y=0 property:
            rhs[k, 0] = 0
    
            # manipulate the k_th row which is responsible for the k-th boundary value
            for j in range(nodes_num):
        
                # matrices for y = 0 property
                system_matrix[k, j] = 1 if k == j else 0
        
        return system_matrix, rhs
    
    
    ################################
    # methods for error estimation #
    ################################
       
    def estimate_error(grid, settings):
        """
        Computes the error indicator 
        eta^2 = sum_i(int_i dt^2 * int_Omega (f - y_d_t - y_d_Delta + p_tt - q_Delta)^2 dx dt)
        for each time intervals and returns an array which contains the error for the i-th row in its ith component.
        
        Notice that the second order p-terms vanish when working with linear finite elements.
        Therefore, the function does not require to get p as an argument
        """
        
        errors = np.zeros(grid.get_num_t_elements())
        
        for e in grid.get_elements():
            
            row_no = e.get_t_row_number()
            dt = e.get_dt()
            
            errors[row_no] += dt**2 * helper.error_integral(e, settings)            
        
        return errors
        
    
    ###############################
    # methods for the integration #
    ###############################
    
    def rhs_integral(k, element, settings):
        """
        Computes the rhs-integral for the fourth order system.
        We use gauss-quadrature for n=4 nodes.
        k denotes the number of the node in the element,
        for which the basis function v_k should assume the value 1
        """
        
        x_min, x_max = element.get_x_bounds()
        t_min, t_max = element.get_t_bounds()
        
        # compute the two factors needed for gauss integration
        dx2 = (x_max - x_min) / 2
        dt2 = (t_max - t_min) / 2
        
        mx2 = (x_min + x_max) / 2
        mt2 = (t_min + t_max) / 2
        
        # get precomputed weights and node positions for gauss integration
        w = helper.gauss_weights
        nodes = helper.gauss_nodes
        
        # start with the integration
        int_val = 0
        
        for i in range(len(w)):
            for j in range(len(w)):
                val = w[i] * w[j]
                
                t = dt2 * nodes[i] + mt2
                x = dx2 * nodes[j] + mx2
                
                val *= settings.f(t, x) - settings.y_d_t(t, x) + settings.get_nu() * settings.y_d_Delta(t, x)
                val *= helper.ref_basis_func(k, 0.5 * nodes[i] + 0.5, 0.5 * nodes[j] + 0.5)
                
                int_val += val
                
        return dx2 * dt2 * int_val
        
    def rhs_y_integral(k, element, settings):
        """
        Computes the rhs-integral for the fourth order system in y.
        We use gauss-quadrature for n=4 nodes.
        k denotes the number of the node in the element,
        for which the basis function v_k should assume the value 1
        """
        
        x_min, x_max = element.get_x_bounds()
        t_min, t_max = element.get_t_bounds()
        
        # compute the two factors needed for gauss integration
        dx2 = (x_max - x_min) / 2
        dt2 = (t_max - t_min) / 2
        
        mx2 = (x_min + x_max) / 2
        mt2 = (t_min + t_max) / 2
        
        # get precomputed weights and node positions for gauss integration
        w = helper.gauss_weights
        nodes = helper.gauss_nodes
        
        # start with the integration
        int_val = 0
        
        for i in range(len(w)):
            for j in range(len(w)):
                val = w[i] * w[j]
                
                t = dt2 * nodes[i] + mt2
                x = dx2 * nodes[j] + mx2
                
                val *= - settings.f_t(t, x) + settings.get_nu() * settings.f_Delta(t, x)
                val *= helper.ref_basis_func(k, 0.5 * nodes[i] + 0.5, 0.5 * nodes[j] + 0.5)
                
                int_val += val
                
        return dx2 * dt2 * int_val
        
    
    def rhs_integral_0(k, element, settings):
        """
        Computes the 't=0'-rhs-integral.
        We use gauss-quadrature for n=4 nodes.
        k denotes the number of the node in the element,
        for which the basis function v_k should assume the value 1
        """
        x_min, x_max = element.get_x_bounds()
        
        # compute the two factors needed for gauss integration
        dx2 = (x_max - x_min) / 2
        mx2 = (x_min + x_max) / 2
        
        # get precomputed weights and node positions for gauss integration
        w = helper.gauss_weights
        nodes = helper.gauss_nodes
        
        # start with the integration
        int_val = 0
        
        for i in range(len(w)):
            val = w[i]
            
            x = dx2 * nodes[i] + mx2
            val *= settings.y_b(x) - settings.y_d(0,x)
            
            val *= helper.ref_basis_func(k, 0, 0.5 * nodes[i] + 0.5)
            
            int_val += val
            
        return dx2 * int_val
        
    def rhs_y_integral_T(k, element, settings):
        """
        Computes the 't=T'-rhs-integral for y.
        We use gauss-quadrature for n=4 nodes.
        k denotes the number of the node in the element,
        for which the basis function v_k should assume the value 1
        """
        x_min, x_max = element.get_x_bounds()
        T = element.get_t_bounds()[1]
        
        # compute the two factors needed for gauss integration
        dx2 = (x_max - x_min) / 2
        mx2 = (x_min + x_max) / 2
        
        # get precomputed weights and node positions for gauss integration
        w = helper.gauss_weights
        nodes = helper.gauss_nodes
        
        # start with the integration
        int_val = 0
        
        for i in range(len(w)):
            val = w[i]
        
            x = dx2 * nodes[i] + mx2
            val *= settings.f(T, x)
        
            val *= helper.ref_basis_func(k, T, 0.5 * nodes[i] + 0.5)
        
            int_val += val
        
        return dx2 * int_val
        
    def rhs_dynamics_integral(k, element, settings):
        """
        Computes the rhs-integral for the dynamics system.
        We use gauss-quadrature for n=4 nodes.
        k denotes the number of the node in the element,
        for which the basis function v_k should assume the value 1
        """
        
        x_min, x_max = element.get_x_bounds()
        t_min, t_max = element.get_t_bounds()
        
        # compute the two factors needed for gauss integration
        dx2 = (x_max - x_min) / 2
        dt2 = (t_max - t_min) / 2
        
        mx2 = (x_min + x_max) / 2
        mt2 = (t_min + t_max) / 2
        
        # get precomputed weights and node positions for gauss integration
        w = helper.gauss_weights
        nodes = helper.gauss_nodes
        
        # start with the integration
        int_val = 0
        
        for i in range(len(w)):
            for j in range(len(w)):
                val = w[i] * w[j]
                val *= settings.f(dt2 * nodes[i] + mt2, dx2 * nodes[j] + mx2)
                val *= helper.ref_basis_func(k, 0.5 * nodes[i] + 0.5, 0.5 * nodes[j] + 0.5)
                
                int_val += val
                
        return dx2 * dt2 * int_val
    
    
    def error_integral(element, settings):
        
        x_min, x_max = element.get_x_bounds()
        t_min, t_max = element.get_t_bounds()
        
        # compute the two factors needed for gauss integration
        dx2 = (x_max - x_min) / 2
        dt2 = (t_max - t_min) / 2
        
        mx2 = (x_min + x_max) / 2
        mt2 = (t_min + t_max) / 2
        
        # get precomputed weights and node positions for gauss integration
        w = helper.gauss_weights
        nodes = helper.gauss_nodes
        
        # start with the integration
        int_val = 0
        
        for i in range(len(w)):
            for j in range(len(w)):
                val = w[i] * w[j]
                
                x_comp = dx2 * nodes[j] + mx2
                t_comp = dt2 * nodes[i] + mt2
                
                val *= (settings.f(t_comp, x_comp) - settings.y_d_t(t_comp, x_comp) + settings.get_nu() * settings.y_d_Delta(t_comp, x_comp))**2
                
                int_val += val
                
        return int_val
    
    
    #############################
    # methods used for plotting #
    #############################
	
    def plot_vec_function(vec, grid, settings, twice_as_long = True, label="", z_min=0, z_max=1, exp_formatter = False):
        """
        Plots i.e. the solution as a surface plot.
        Requires either the solution vector [p,q] in vectorized form (ordering as usual), or just a vecor p.
        """
         
        # only need the first half for the solution as long as not stated otherwise
        if twice_as_long:
            vec = vec[:vec.shape[0] // 2]
        
        if vec.shape[1] != 1:
            sys.exit("ERROR: Plot-method has to get functions in vector shape!")
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(9,8))
        
        if vec.shape[0] != grid.get_num_nodes():
            sys.exit(f"ERROR: Shape of vector and plot does not match! Is: {vec.shape}, must be: {(grid.get_num_nodes(), 1)}")
        
        # Make data.
        X = grid.get_x_space()
        T = grid.get_t_space()
        X, T = np.meshgrid(X, T)
        
        Z = vec.reshape(X.shape)

        # Plot the surface.
        surf = ax.plot_surface(X, T, Z,
                            cmap=cm.coolwarm,
                            linewidth=0,
                            antialiased=False)


        #ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.1e}' if exp_formatter else'{x:.02f}')
        
        ax.tick_params(axis='z', which='major', pad=15, labelsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        ax.set_xlabel('$x$', fontsize=15)
        ax.set_ylabel('$t$', fontsize=15)
        
        plt.locator_params(nbins=3)
        ax.set_zlim(z_min, z_max)
        
        ax.view_init(30, -20, 0)
        
        plt.locator_params(axis='z', nbins=5)
        
        
    def plot_control(vec, grid, settings):
        """
        Plots the control in a common plot with the correct initial condition.
        Requires the [p,q] values to be given in vectorized form. Ordering as usual.
        """
        
        x_space = grid.get_x_space()
        
        # the t=0 nodes are the first nodes in the row
        p_0 = vec[:len(x_space)]
        
        # background guess
        y_b = settings.y_b(x_space).reshape((len(x_space), 1))
        
        # true solution is y_d for t=0
        y_true = settings.y_d(0, x_space).reshape((len(x_space), 1))
        
        # the control can be computed with y_b and p(0)
        control = y_b - 1.0/settings.get_alpha() * p_0
        
        plt.figure()
        
        # plot all of them
        plt.plot(x_space, control, label="control")
        plt.plot(x_space, y_true, label="reality")
        plt.plot(x_space, y_b, label="previous prediction")
        
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.locator_params(nbins=3)
        
        plt.legend()
        
    def plot_heatmap_error(vec1, vec2, grid, v_max = 1, v_min=0):
        """
        Plots the error between two vectors vec1 and vec2 as a heatmap.
        The vectors must have a length which is equal to the total number of points. 
        """
        
        if vec1.shape != vec2.shape:
            sys.exit("ERROR: Incompatible shape!")
        
        # compute the error and reshape the data
        error = np.abs(vec1 - vec2).reshape((grid.get_num_t_nodes(), grid.get_num_x_nodes()))
        
        # style the figure
        fig, ax = plt.subplots()
        plt.imshow(error, cmap='hot', interpolation='nearest', origin='lower', vmax=v_max, vmin=v_min)
        
        plt.xlabel('$x$', fontsize=15)
        plt.ylabel('$t$', fontsize=15, rotation="horizontal")
        
        plt.xticks([0, int(grid.get_num_x_nodes() / 2), grid.get_num_x_nodes() - 1], [0, 0.5, 1.0], fontsize=15)
        plt.yticks([0, int(grid.get_num_t_nodes() / 2), grid.get_num_t_nodes() - 1], [0, 0.5, 1.0], fontsize=15)
        plt.locator_params(nbins=3)
        
        largest = (0,0)
        largest_err = 0.0
        # additionally mark the largest point:
        for i in range(grid.get_num_t_nodes()):
            for j in range(grid.get_num_x_nodes()):
                if error[i,j] > largest_err:
                    largest = (i,j)
                    largest_err = error[i,j]
                    
        plt.scatter(largest[1], largest[0], c="red")
        
        
        print(f"largest error: {largest_err}")
        ax.text(largest[1] + 2, largest[0] - 0.7, "$e_\\text{max}$",
                      size=15,
                      color='red')
                      
        
    def compare_controls(controls, settings, grid, labels=[], xvals=[]):
        
        x_space = grid.get_x_space()
        
        plt.figure(figsize=(9,9))
        plt.xlim(*settings.get_bounds_x())
        
        c1 = (0, 0, 1)
        c2 = (0, 0.5, 0)
        
        # true solution is y_d for t=0
        y_true = settings.y_d(0, x_space).reshape((len(x_space), 1))
        plt.plot(x_space, y_true, label="$y^{(d)}(0)$ (reality)", color=c1, linewidth=2)
        
        # background guess
        y_background = settings.y_b(x_space).reshape((len(x_space), 1))
        plt.plot(x_space, y_background, label="$y^{(b)}$ (bakground guess)", color=c2, linewidth=2)
        
        num = len(controls)
        
        min_val, max_val = 0, 1
        
        for li, control in enumerate(controls):
            
            # determine color
            c = tuple(map(lambda i, j: (1 - li / num) * i + (li / num) * j, c1, c2))
            
            # plot all of them
            plt.plot(x_space, control, label=labels[li], color=c, linewidth=1, linestyle="dashed")
            
            min_val = min(min(control), min_val)
            max_val = max(max(control), max_val)
          
        # position of the labels on the line
        labelLines(plt.gca().get_lines(), zorder=2.5, xvals=xvals, align=True, fontsize=15)
        
        x_min, x_max = settings.get_bounds_x()
        plt.gca().set_aspect((x_max - x_min)/(max_val - min_val))
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.locator_params(nbins=3)
        
    def compare_controls_same_initial(controls, settings, grid, labels=[], xvals=[]):
        
        x_space = grid.get_x_space()
        
        plt.figure(figsize=(9,9))
        plt.xlim(*settings.get_bounds_x())
        
        c1 = (0, 0, 1)
        c2 = (0, 0.5, 0)
        
        # true solution is y_d for t=0
        y_true = settings.y_d(0, x_space).reshape((len(x_space), 1))
        plt.plot(x_space, y_true, label="$y^{(d)}(0)$ (reality)", color=c1, linewidth=2)
        
        # background guess
        y_background = settings.y_b(x_space).reshape((len(x_space), 1))
        plt.plot(x_space, y_background, label="$y^{(b)}$ (bakground guess)", color=c2, linewidth=2, linestyle="dashed")
        
        num = len(controls)
        
        min_val, max_val = 0, 1
        
        for li, control in enumerate(controls):
            
            # determine color
            c = tuple(map(lambda i, j: (1 - li / num) * i + (li / num) * j, c1, c2))
            
            # plot all of them
            plt.plot(x_space, control, label=labels[li], color="red", linewidth=1)
            
            min_val = min(min(control), min_val)
            max_val = max(max(control), max_val)
          
        # position of the labels on the line
        labelLines(plt.gca().get_lines(), zorder=2.5, xvals=xvals, align=True, fontsize=15)
        
        x_min, x_max = settings.get_bounds_x()
        plt.gca().set_aspect((x_max - x_min)/(max_val - min_val))
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.locator_params(nbins=3)
        