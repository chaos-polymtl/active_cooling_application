import numpy as np
from scipy.sparse.linalg import bicgstab, gmres, spilu, LinearOperator, spsolve
from scipy.sparse import lil_matrix, csr_matrix
from src.boundary import Boundary

from src.utility import *

class FiniteDifferenceSolver:
    def __init__(self, params, time_manager):
        """
        Initialize the FiniteDifferenceSolver.

        :param params: Parameters object
        :param time_manager: Object managing all time-related variables (start_time, time_step, final_time)

        """
        self.params = params
        self.time_manager = time_manager

        self.T = np.repeat(params.initial_condition, params.nx * params.ny * params.nz)
        self.rhs = self.T.copy()

        self.A = lil_matrix((params.nx*params.ny*params.nz, params.nx*params.ny*params.nz))

        # Create the grid points
        x = np.linspace(0, self.params.plate_dimensions[0], params.nx)
        y = np.linspace(0, self.params.plate_dimensions[1], params.ny)
        z = np.linspace(0, self.params.plate_dimensions[2], params.nz)
        self.points = np.array([[xi, yi, zi] for zi in z for yi in y for xi in x])

        # Element size
        self.dx = params.plate_dimensions[0]/(params.nx-1)
        self.dy = params.plate_dimensions[1]/(params.ny-1)
        self.dz = params.plate_dimensions[2]/(params.nz-1)

        self.boundary = Boundary(params, self.points, self.dx, self.dy, self.dz)

        # Constant solid properties
        self.alpha = params.thermal_conductivity/(params.density*params.heat_capacity) # k/rho*Cp

        self.calculate_fourrier_numbers()

        self.heat_flux = np.zeros((params.nx*params.ny*params.nz, 3))

    def calculate_fourrier_numbers(self):
        """
        Calculate the fourrier numbers for the finite difference model alpha * dt / (dx ** 2).
        """

        self.fourrier_x = self.alpha*self.time_manager.time_step/np.power(self.dx, 2)
        self.fourrier_y = self.alpha*self.time_manager.time_step/np.power(self.dy, 2)
        self.fourrier_z = self.alpha*self.time_manager.time_step/np.power(self.dz, 2)

    def solve(self, return_solution=False, sensitivity=False, direct_temperature=None, d_k=None, h_map=None):
        """
        Solve the finite difference model at every time specified by the time manager.
        
        :return: solution at the final time step
        """

        # Assemble the A matrix in Ax = b
        self.assemble_matrix()

        # Assemble the right-hand side (b) in Ax = b
        self.assemble_rhs()

        # Assemble the boundary conditions
        self.boundary.assemble_boundaries(self.A, self.rhs, self.time_manager.current_time, sensitivity, direct_temperature, d_k)

        # Convert matrix to more efficient sparsity pattern
        self.A = self.A.tocsc()

        # Solve the linear system (A*x = b)
        self.T = self.solve_linear_system(self.A, self.rhs)

        # Compute the heat flux
        self.compute_flux()

    def assemble_matrix(self):
        """
        Assemble the matrix finite difference model.
        """
        nx, ny, nz = self.params.nx, self.params.ny, self.params.nz

        self.A = lil_matrix((nx*ny*nz, nx*ny*nz))
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz - 1):
                    index = flatten_index(i, j, k, nx, ny)

                    # Internal points
                    self.A[index, index] = 1 + 2*self.fourrier_x + 2*self.fourrier_y + 2*self.fourrier_z
                    self.A[index, index-1] = -self.fourrier_x
                    self.A[index, index+1] = -self.fourrier_x
                    self.A[index, index-nx] = -self.fourrier_y
                    self.A[index, index+nx] = -self.fourrier_y
                    self.A[index, index-nx*ny] = -self.fourrier_z
                    self.A[index, index+nx*ny] = -self.fourrier_z

    def assemble_rhs(self):
        """
        Assemble the right-hand side of the linear system.
        """
        self.rhs = self.T.copy()

        if self.params.source:
            self.evaluate_source_term()

    def solve_linear_system(self, A, rhs, function=gmres):
        """
        Solve the linear system A*x = rhs.
        """
        ilu_preconditioner = spilu(A)
        M = LinearOperator(A.shape, ilu_preconditioner.solve)
        solution, _ = function(A, rhs, x0=self.T, M=M, rtol=1e-9)

        return solution
    
    def evaluate_source_term(self):
        """
        Evaluate source term.
        """

        nx, ny, nz = self.params.nx, self.params.ny, self.params.nz

        self.source_term = np.zeros(nx*ny*nz)

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    index = flatten_index(i, j, k, nx, ny)
                    x, y, z = self.points[index]
                    self.source_term[index] = eval(self.params.source_term)
                    self.rhs[index] -= self.alpha * self.source_term[index] * self.time_manager.time_step
    
    def compute_flux(self):
        """
        Compute the heat flux.
        """
        nx, ny, nz = self.params.nx, self.params.ny, self.params.nz
        k = self.params.thermal_conductivity

        # Reshape temperature array for easy indexing
        T = self.T.reshape((nx, ny, nz), order="F")

        grad_x = np.zeros_like(T)
        grad_y = np.zeros_like(T)
        grad_z = np.zeros_like(T)

        # Interior nodes
        grad_x[1:-1, :, :] = (T[2:, :, :] - T[:-2, :, :]) / (2 * self.dx)
        grad_y[:, 1:-1, :] = (T[:, 2:, :] - T[:, :-2, :]) / (2 * self.dy)
        grad_z[:, :, 1:-1] = (T[:, :, 2:] - T[:, :, :-2]) / (2 * self.dz)

        # Boundaries
        grad_x[0, :, :] = (-T[2, :, :] + 4*T[1, :, :] - 3*T[0, :, :]) / (2 * self.dx)
        grad_x[-1, :, :] = (T[-3, :, :] - 4*T[-2, :, :] + 3*T[-1, :, :]) / (2 * self.dx)

        grad_y[:, 0, :] = (-T[:, 2, :] + 4*T[:, 1, :] - 3*T[:, 0, :]) / (2 * self.dy)
        grad_y[:, -1, :] = (T[:, -3, :] - 4*T[:, -2, :] + 3*T[:, -1, :]) / (2 * self.dy)

        grad_z[:, :, 0] = (-T[:, :, 2] + 4*T[:, :, 1] - 3*T[:, :, 0]) / (2 * self.dz)
        grad_z[:, :, -1] = (T[:, :, -3] - 4*T[:, :, -2] + 3*T[:, :, -1]) / (2 * self.dz)

        # Compute flux
        flux_x = -k * grad_x
        flux_y = -k * grad_y
        flux_z = -k * grad_z

        # Reshape the heat flux into a vector
        self.heat_flux = np.stack([flux_x, flux_y, flux_z], axis=-1).reshape(-1, 3, order="F")

    def get_temperature_face(self, face_id, values=None):
        '''Getter for temperature at the face of the plate'''
        if values is None:
            return self.T[self.boundary.dict_boundary_points[face_id]]
        else:
            return values[self.boundary.dict_boundary_points[face_id]]


class FiniteDifferenceSolverSS:
    def __init__(self, params, initial_finite_difference = None):
        """
        Initialize the FiniteDifferenceSolverSS.

        :param params: Parameters object

        """
        print("Initializing steady-state solver...")
        self.params = params

        self.T = initial_finite_difference.T if initial_finite_difference else np.repeat(params.initial_condition, params.nx * params.ny * params.nz)
        self.rhs = self.T.copy()

        self.A = lil_matrix((params.nx*params.ny*params.nz, params.nx*params.ny*params.nz))

        # Create the grid points
        x = np.linspace(0, self.params.plate_dimensions[0], params.nx)
        y = np.linspace(0, self.params.plate_dimensions[1], params.ny)
        z = np.linspace(0, self.params.plate_dimensions[2], params.nz)
        self.points = np.array([[xi, yi, zi] for zi in z for yi in y for xi in x])

        # Element size
        self.dx = params.plate_dimensions[0]/(params.nx-1)
        self.dy = params.plate_dimensions[1]/(params.ny-1)
        self.dz = params.plate_dimensions[2]/(params.nz-1)

        self.boundary = Boundary(params, self.points, self.dx, self.dy, self.dz)

        # Constant solid properties
        self.alpha = params.thermal_conductivity/(params.density*params.heat_capacity) # k/rho*Cp

        self.heat_flux = np.zeros((params.nx*params.ny*params.nz, 3))

    def reset_T_and_rhs(self):
        """
        Reset the temperature and right-hand side vectors to their initial values.
        """
        self.T = np.repeat(self.params.initial_condition, self.params.nx * self.params.ny * self.params.nz)
        self.rhs = self.T.copy()

    def compute_flux(self):
        """
        Compute the heat flux.
        """
        nx, ny, nz = self.params.nx, self.params.ny, self.params.nz
        k = self.params.thermal_conductivity

        # Reshape temperature array for easy indexing
        T = self.T.reshape((nx, ny, nz), order="F")

        grad_x = np.zeros_like(T)
        grad_y = np.zeros_like(T)
        grad_z = np.zeros_like(T)

        # Interior nodes
        grad_x[1:-1, :, :] = (T[2:, :, :] - T[:-2, :, :]) / (2 * self.dx)
        grad_y[:, 1:-1, :] = (T[:, 2:, :] - T[:, :-2, :]) / (2 * self.dy)
        grad_z[:, :, 1:-1] = (T[:, :, 2:] - T[:, :, :-2]) / (2 * self.dz)

        # Boundaries
        grad_x[0, :, :] = (-T[2, :, :] + 4*T[1, :, :] - 3*T[0, :, :]) / (2 * self.dx)
        grad_x[-1, :, :] = (T[-3, :, :] - 4*T[-2, :, :] + 3*T[-1, :, :]) / (2 * self.dx)

        grad_y[:, 0, :] = (-T[:, 2, :] + 4*T[:, 1, :] - 3*T[:, 0, :]) / (2 * self.dy)
        grad_y[:, -1, :] = (T[:, -3, :] - 4*T[:, -2, :] + 3*T[:, -1, :]) / (2 * self.dy)

        grad_z[:, :, 0] = (-T[:, :, 2] + 4*T[:, :, 1] - 3*T[:, :, 0]) / (2 * self.dz)
        grad_z[:, :, -1] = (T[:, :, -3] - 4*T[:, :, -2] + 3*T[:, :, -1]) / (2 * self.dz)

        # Compute flux
        flux_x = -k * grad_x
        flux_y = -k * grad_y
        flux_z = -k * grad_z

        # Reshape the heat flux into a vector
        self.heat_flux = np.stack([flux_x, flux_y, flux_z], axis=-1).reshape(-1, 3, order="F")

    def solve(self, transpose=False):
        """
        Solve the steady-state finite difference model (no time dependence).
        """
        # Assemble the boundary conditions for steady-state
        if transpose == False:
            self.assemble_matrix()
            self.assemble_rhs()
            self.boundary.assemble_boundaries_ss(self.A, self.rhs)

        # Solve the linear system (A*x = b)
        self.T = self.solve_linear_system(self.A, self.rhs)

        self.compute_flux()

    def assemble_matrix(self):
        """
        Assemble the matrix for the steady-state finite difference model.
        """
        nx, ny, nz = self.params.nx, self.params.ny, self.params.nz
        alpha_x = self.params.thermal_conductivity / self.dx**2
        alpha_y = self.params.thermal_conductivity / self.dy**2
        alpha_z = self.params.thermal_conductivity / self.dz**2

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    index = flatten_index(i, j, k, nx, ny)

                    # Internal points
                    self.A[index, index] = 2 * (alpha_x + alpha_y + alpha_z)
                    self.A[index, index - 1] = -alpha_x
                    self.A[index, index + 1] = -alpha_x
                    self.A[index, index - nx] = -alpha_y
                    self.A[index, index + nx] = -alpha_y
                    self.A[index, index - nx * ny] = -alpha_z
                    self.A[index, index + nx * ny] = -alpha_z

    def assemble_rhs(self):
        """
        Assemble the right-hand side of the linear system for steady-state.
        """
        nx, ny, nz = self.params.nx, self.params.ny, self.params.nz

        self.rhs = np.zeros(nx * ny * nz)

        if self.params.source:
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    for k in range(1, nz - 1):
                        index = flatten_index(i, j, k, nx, ny)
                        self.rhs[index] = eval(self.params.source_term)

    def solve_linear_system(self, A, rhs):
        """
        Solve the linear system A*x = rhs.
        """

        A_csr = csr_matrix(A)

        solution = spsolve(A_csr, rhs)

        return solution

    def jacobi_preconditioner(self):
        """
        Create a Jacobi preconditioner for a given sparse matrix A.
        
        Parameters:
        A (scipy.sparse.spmatrix): Sparse matrix for which to compute the preconditioner.

        Returns:
        LinearOperator: A linear operator that represents the Jacobi preconditioner.
        """
        # Extract the diagonal
        diag = self.A.diagonal()
        
        # Replace zeros in the diagonal to avoid division by zero
        diag[diag == 0] = 1.0

        # Create a function to apply the preconditioner
        def apply_preconditioner(x):
            return x / diag

        # Return the preconditioner as a LinearOperator
        return LinearOperator(self.A.shape, matvec=apply_preconditioner)
    
    def get_temperature_face(self, face_id, values=None):
        '''Getter for temperature at the face of the plate'''
        if values is None:
            return self.T[self.boundary.dict_boundary_points[face_id]]
        else:
            return values[self.boundary.dict_boundary_points[face_id]]
        
class AdjointSolver:
    def __init__(self, params, time_manager, target_temperature):
        """
        Initialize the AdjointSolver.

        :param params: Parameters object
        :param time_manager: Object managing all time-related variables (start_time, time_step, final_time)
        :param target_temperature: Target temperature mesured at the lab for the adjoint solver

        """
        
        self.params = params
        self.time_manager = time_manager
        self.target_temperature = target_temperature

        # initial condition at 0 at the initial time
        self.lambda_t = np.repeat(0., params.nx * params.ny * params.nz)
        self.rhs = self.lambda_t.copy()

        self.A = lil_matrix((params.nx*params.ny*params.nz, params.nx*params.ny*params.nz))

        # Create the grid points
        x = np.linspace(0, self.params.plate_dimensions[0], params.nx)
        y = np.linspace(0, self.params.plate_dimensions[1], params.ny)
        z = np.linspace(0, self.params.plate_dimensions[2], params.nz)
        self.points = np.array([[xi, yi, zi] for zi in z for yi in y for xi in x])

        # Element size
        self.dx = params.plate_dimensions[0]/(params.nx-1)
        self.dy = params.plate_dimensions[1]/(params.ny-1)
        self.dz = params.plate_dimensions[2]/(params.nz-1)

        self.boundary = Boundary(params, self.points, self.dx, self.dy, self.dz)
        self.dict_boundary_points = compute_boundary_dict(self.params.nx, self.params.ny, self.params.nz)

        # Constant solid properties
        self.alpha = params.thermal_conductivity/(params.density*params.heat_capacity) # k/rho*Cp

        self.calculate_fourrier_numbers()

        self.heat_flux = np.zeros((params.nx*params.ny*params.nz, 3))

    def calculate_fourrier_numbers(self):
        """
        Calculate the fourrier numbers for the finite difference model alpha * dt / (dx ** 2).
        """

        self.fourrier_x = self.alpha*self.time_manager.time_step/np.power(self.dx, 2)
        self.fourrier_y = self.alpha*self.time_manager.time_step/np.power(self.dy, 2)
        self.fourrier_z = self.alpha*self.time_manager.time_step/np.power(self.dz, 2)

    def solve(self, direct_solution, target_temperature):
        """
        Solve the finite difference model at every time specified by the time manager.
        
        :param direct_solution: solution at the current time step of the direct solver.
        """

        # Assemble the A matrix in Ax = b
        self.assemble_matrix()

        # Assemble the right-hand side (b) in Ax = b
        self.assemble_rhs()

        # Assemble the boundary conditions
        self.boundary.assemble_boundaries_adjoint(self.A, self.rhs, self.time_manager.current_time)

        # Apply adjoint source term
        self.assemble_adjoint_source_term(direct_solution, target_temperature)

        # Convert matrix to more efficient sparsity pattern
        self.A = self.A.tocsc()
       
        self.lambda_t = self.solve_linear_system(self.A, self.rhs)

    def assemble_matrix(self):
        """
        Assemble the matrix finite difference model.
        """
        nx, ny, nz = self.params.nx, self.params.ny, self.params.nz

        self.A = lil_matrix((nx*ny*nz, nx*ny*nz))
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz - 1):
                    index = flatten_index(i, j, k, nx, ny)

                    # Internal points
                    self.A[index, index] = 1 + 2*self.fourrier_x + 2*self.fourrier_y + 2*self.fourrier_z #changement de signe pour l'adjoint
                    self.A[index, index-1] = -self.fourrier_x
                    self.A[index, index+1] = -self.fourrier_x
                    self.A[index, index-nx] = -self.fourrier_y
                    self.A[index, index+nx] = -self.fourrier_y
                    self.A[index, index-nx*ny] = -self.fourrier_z
                    self.A[index, index+nx*ny] = -self.fourrier_z

    def assemble_rhs(self):
        """
        Assemble the right-hand side of the linear system. No source term allowed for now
        """
        self.rhs = self.lambda_t.copy()

    def solve_linear_system(self, A, rhs, function=gmres):
        """
        Solve the linear system A*x = rhs.
        """
        ilu_preconditioner = spilu(A)
        M = LinearOperator(A.shape, ilu_preconditioner.solve)
        solution, _ = function(A, rhs, x0=self.lambda_t, M=M, rtol=1e-9)

        return solution

    def assemble_adjoint_source_term(self, direct_solution, target_temperature):
        """
         In the adjoint problem, a source term that depends on the mesured temperature and the direct solution is added on the nodes of the adjoint target face. 
        
        :param direct_solution: solution at the current time step of the direct solver.
        :param target_temperature: temperature measured experimentally at time t-dt step of the adjoint solver.
        """
        # Target temperature at the current time step
        for i, index in enumerate(self.dict_boundary_points[self.params.adjoint_target_face]): 
            self.rhs[index] += 2*self.time_manager.time_step*(direct_solution[i] - target_temperature[i])
    
    def get_lambda_at_face(self, face_id):
        '''Getter for lambda at the face of the plate'''
        return self.lambda_t[self.boundary.dict_boundary_points[face_id]]