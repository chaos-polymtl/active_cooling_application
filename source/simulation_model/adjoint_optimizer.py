import os
import numpy as np
import csv
from scipy.ndimage import gaussian_filter

import copy
from scipy.linalg import solve
from src.finite_difference_3d import FiniteDifferenceSolverSS
from src.finite_difference_3d import FiniteDifferenceSolver
from src.finite_difference_3d import AdjointSolver
from scipy.optimize import minimize

from src.time_manager import TimeManager

class AdjointTransient:
    def __init__(self, params, finite_difference, data_manager, target_snapshots=None):
        """
        Initialize the transient adjoint optimizer
        :param params: Parameters object containing simulation parameters
        :param finite_difference: FiniteDifferenceSolver object for the direct problem (to remove in the future)
        :param data_manager: DataManager object for saving results
        :param target_snapshots: list or tuple of temperature fields [T_prev, T_curr]
        representing consecutive system states for the transient reconstruction.
        """

        self.params = params
        self.finite_difference = finite_difference
        self.data_manager = data_manager
        self.target_snapshots = target_snapshots

        if hasattr(self.params, 'adjoint_target_array'):
            # If we are in transient mode, we should have a a target array for every time step -
            # > for 1 time step we have a np.array of size (, nx*ny). 
            # for 4 time steps we would have a np.array of size (4, nx*ny). 
            # The dimension of the array is taken into acount in the adjoint solver
            
            self.initial_temp = self.params.adjoint_target_array[0]
            
            # get the target number of time steps
            n_time_steps = int((self.params.final_time - self.params.start_time)/self.params.time_step)
            
            self.target_T = []
            for i in range(n_time_steps):
                # Store the target temperature for each time step (excluding initial condition at index 0)
                self.target_T.append(self.params.adjoint_target_array[i+1])
        
        elif target_snapshots is not None:
            # --- MPC transient mode: two-snapshot input ---
            if len(target_snapshots) != 2:
                raise ValueError("Expected two snapshots [T_prev, T_curr] for transient reconstruction.")
            self.initial_temp = np.asarray(target_snapshots[0])
            self.target_T = [np.asarray(target_snapshots[1])]
    
        else:
            # Error if we do not have a file that points to the target temperature. maybe put this in the params file in the future
            raise ValueError("Please provide a target temperature array or two snapshots for the unsteady adjoint problem.")

        self.face_to_optimize = self.params.adjoint_optimize_face

    def run_nonlinear(self, return_h=False):
        """
        Run the nonlinear adjoint optimization loop
        
        :param return_h: Boolean indicating whether to return the optimized convective coefficient array instead of saving to file
        """

        # Handle missing DataManager (for MPC integration, no VTU output needed)
        if self.data_manager is None:
            def _noop_save_vtu(*args, **kwargs):
                pass
            self.data_manager = type("DummyDM", (), {"save_vtu": staticmethod(_noop_save_vtu)})()

        # Check if we are in 2-snapshot transient mode (MPC)
        if hasattr(self, 'target_T') and len(self.target_T) == 1 and hasattr(self, 'initial_temp'):
            n_time_steps = 1
        else:
            n_time_steps = int((self.params.final_time - self.params.start_time)/self.params.time_step)

        # Setup adjoint loop
        error = 1e6
        tolerance = self.params.adjoint_tolerance
        
        iteration = 0
        max_iteration = self.params.adjoint_max_iterations

        target_face = self.params.adjoint_target_face
        face_to_optimize = self.params.adjoint_optimize_face

        n_points = self.params.nx * self.params.ny
        direct_temperature_solutions = np.zeros((n_time_steps, n_points))
        adjoint_temperature_solutions = np.zeros((n_time_steps, n_points))
        gradient = np.zeros((n_time_steps, n_points))
        previous_gradient = np.zeros((n_time_steps, n_points))
        h_new = np.zeros((n_time_steps, n_points))

        moment = 0
        moment_2 = 0

        beta_1 = 0.9
        beta_2 = 0.999

        # Even if the h coefficient is only on boundaries, we give the value of zero to all the other points just for the output. 
        # The last convective coefficient is always 0 because the h at the current time step is used to calculate the temperature for the next time step.
        h_coeff_to_output = np.zeros(n_points*self.params.nz)
        temp_error_to_output = np.zeros(n_points*self.params.nz)

        # Set all the convective coefficients 
        for i in range(n_time_steps):
            h_new[i] = self.finite_difference.boundary.get_convective_coefficient_at_face(face_to_optimize)

        while error > tolerance and iteration < max_iteration:

            # Restart the time manager at every iteration
            time_manager = TimeManager(self.params)
            self.finite_difference = FiniteDifferenceSolver(self.params, time_manager)
            time_step = 0
            # Set the initial condition for the direct problem - TODO INTERPOLATE THE INITIAL CONDITION
            if hasattr(self, "initial_temp"):
                # 2-snapshot mode
                init_scalar = float(np.average(self.initial_temp))
            else:
                # full transient mode
                init_scalar = float(np.average(self.params.adjoint_target_array[0]))

            self.finite_difference.T = np.full(self.params.nx * self.params.ny * self.params.nz, init_scalar, dtype=float)

            while not time_manager.is_finished() and time_step < n_time_steps:
                idx = max(0, min(time_step, n_time_steps - 1))  

                # Set the h coefficient to the one at the current time step
                self.finite_difference.boundary.set_convective_heat_transfer_map_for_face(face_to_optimize, h_new[idx])
                
                # Prepare to output the h coefficient
                id_h_coefficient_imposed = self.finite_difference.boundary.convective_coefficient.keys()
                for id in id_h_coefficient_imposed:
                    h_coeff_to_output[id] = self.finite_difference.boundary.convective_coefficient[id]

                # Compute the difference between the temperature at the current time step and the target temperature
                if time_step != 0:
                    temp_error_on_face = self.finite_difference.get_temperature_face(target_face) - self.target_T[time_step-1]
                else:
                    temp_error_on_face = self.finite_difference.get_temperature_face(target_face)*0.

                for i, index in enumerate(self.finite_difference.boundary.dict_boundary_points[target_face]):
                    temp_error_to_output[index] = temp_error_on_face[i]

                # We save the solution at every time step. The first one is the initial condition
                self.data_manager.save_vtu(time_manager.current_step, self.finite_difference.T, self.finite_difference.heat_flux, h_coeff_to_output, temp_error_to_output)

                # Solve the unsteady direct problem for the current time step
                self.finite_difference.solve()

                # Store the temperature on the top face in the solution vector. IMPORTANT - even if the temperature solution is calculated for the time t+time_step, we store the direct temperature solution on the face at time t. This is done for the adjoint problem
                direct_temperature_solutions[time_step] = self.finite_difference.get_temperature_face(target_face) 

                time_manager.update_time()
                time_step += 1

            # need to compute the error if only two time steps
            if time_step != 0:
                temp_error_on_face = self.finite_difference.get_temperature_face(target_face) - self.target_T[time_step-1]
            else:
                temp_error_on_face = self.finite_difference.get_temperature_face(target_face)*0.

            for i, index in enumerate(self.finite_difference.boundary.dict_boundary_points[target_face]):
                temp_error_to_output[index] = temp_error_on_face[i]

            # save the final solution once we exit the loop. The last h_coeff_to_output is the same as one step before the end of the simulation.
            self.data_manager.save_vtu(time_manager.current_step, self.finite_difference.T, self.finite_difference.heat_flux, h_coeff_to_output, temp_error_to_output)

            # We go back in time solving the adjoint problem. For that we restart the time manager (going forward in time does not change the equation we solve)
            time_manager = TimeManager(self.params)
            self.adjoint = AdjointSolver(self.params, time_manager, self.target_T) #The shape of target_T depends on the number of time steps
            while not time_manager.is_finished() and time_step > 0:
                idx = max(0, min(time_step - 1, n_time_steps - 1))
                
                # Set the h coefficient to the one at the next time step (which is the previous one in the adjoint problem)
                self.adjoint.boundary.set_convective_heat_transfer_map_for_face(face_to_optimize, h_new[idx])

                # Prepare the heat transfer coefficient for output
                id_h_coefficient_imposed = self.adjoint.boundary.convective_coefficient.keys()
                for id in id_h_coefficient_imposed:
                    h_coeff_to_output[id] = self.adjoint.boundary.convective_coefficient[id]

                self.data_manager.save_vtu(time_step, self.adjoint.lambda_t, self.finite_difference.heat_flux, h_coeff_to_output, file_name_prefix="adjoint_output")

                # Solve the adjoint problem using the direct problem solution
                self.adjoint.solve(direct_temperature_solutions[time_step-1], self.target_T[time_step-1])
                
                # Store the adjoint temperature on the top face in the solution vector
                adjoint_temperature_solutions[time_step - 1] = self.adjoint.get_lambda_at_face(target_face)

                # Update the time
                time_manager.update_time()
                time_step -= 1
                

            # We save the final solution, which is at the end
            self.data_manager.save_vtu(time_step, self.adjoint.lambda_t, self.finite_difference.heat_flux, h_coeff_to_output, file_name_prefix="adjoint_output")

            # Calculate the gradient of the cost function to minimize using the solution of the adjoint and the direct problem
            gradient = -1/(self.params.thermal_conductivity) * adjoint_temperature_solutions * (direct_temperature_solutions - eval(self.params.T_inf[target_face]))

            h_previous = h_new.copy()

            # Gradient descent for now -> method to be improved with maths in master thesis
            h_new = h_previous - 0.0001*gradient

            # Apply constraint
            h_new = np.maximum(h_new, 0.0)
                        
            # Calculate the error as RMSE
            error = np.linalg.norm(direct_temperature_solutions - self.target_T, 2)/np.sqrt(len(self.target_T)*len(self.target_T[0]))
            
            print("error: ", round(error, 8))

            iteration += 1

        # write the h_new vector to a file
        np.savetxt("reconstructed_h.csv", h_new, delimiter=",")

        if return_h:
            return np.asarray(h_new, dtype=float)

class AdjointSS:
    def __init__(self, params, finite_difference, data_manager):
        self.finite_difference_ss = finite_difference
        self.params = params
        self.data_manager = data_manager
        
        if hasattr(self.params, 'adjoint_target_array'):
            self.target_T = self.params.adjoint_target_array
        else:
            self.target_T = np.ones_like(self.finite_difference_ss.T) * self.target_T

        self.target_face = self.params.adjoint_target_face
        self.optimize_face = self.params.adjoint_optimize_face


    def run_nonlinear(self):
        # Adjust adjoint boundary parameters
        self.params_adjoint = copy.deepcopy(self.params)
        target_face = self.params_adjoint.adjoint_target_face

        # Setup adjoint loop
        error = 1e6
        error_old = 1e6
        tolerance = self.params_adjoint.adjoint_tolerance
        max_iterations = self.params_adjoint.adjoint_max_iterations
        iteration = 0
        optimize_face = self.params_adjoint.adjoint_optimize_face
        self.finite_difference_ss.solve()
        current_T = self.finite_difference_ss.get_temperature_face(target_face)
        lamb = np.zeros_like(self.finite_difference_ss.T)
        step_size = self.params.adjoint_step_size
        moment = 0
        moment_2 = 0
        beta = 0
        beta_2 = 0

        # Loop until target found
        while error > tolerance and iteration < max_iterations:

            # Calculate the dirichlet boundary condition
            adjoint_rhs = current_T - self.target_T

            # Solve the adjoint problem
            self.adjoint = copy.deepcopy(self.finite_difference_ss)
            self.adjoint.boundary.set_rhs(target_face, self.adjoint.rhs, -adjoint_rhs)
            # self.adjoint.boundary.set_A(target_face, self.adjoint.A, 1.0)
            self.adjoint.solve(transpose=True)

            # Store lambda transpose and apply the optimization step
            lamb = self.adjoint.get_temperature_face(optimize_face, values = self.adjoint.T)

            # Calculate gradient
            old_h = self.finite_difference_ss.boundary.get_convective_coefficient_at_face(optimize_face)
            
            gradient = -lamb * (current_T - float(self.params.T_inf[target_face]))

            # Calculate momentum
            
            moment = beta * moment + (1 - beta) * gradient
            moment_2 = beta_2 * moment_2 + (1 - beta_2) * np.square(gradient)

            # Apply newton method
            new_h = old_h + 10* moment / np.sqrt(moment_2 + 1e-8)

            # Apply constraint
            new_h = np.maximum(new_h, 0.0)

            # Apply new convection coefficient
            self.finite_difference_ss.boundary.set_convective_heat_transfer_map_for_face(optimize_face, new_h)

            # Run steady state simulation
            self.finite_difference_ss.solve()
            if self.params_adjoint.save_output:
                self.data_manager.save_vtu(1, self.finite_difference_ss.T, self.finite_difference_ss.heat_flux)
                self.data_manager.generate_pvd(1)

            # Calculate error as RMSE
            new_T = self.finite_difference_ss.get_temperature_face(target_face)
            error = np.linalg.norm(new_T - self.target_T, 2)/np.sqrt(len(new_T))
            current_T = new_T

            # Print logs
            print(f'Iteration: {iteration}, Error: {error:.4f}')

            # Save adjoint output
            if self.params.adjoint_output_results:
                self.save_adjoint_output(iteration, error, new_h, current_T, "adjoint_output")

            beta = 0.9
            beta_2 = 0.999

            iteration += 1

        print(f'Iterations: {iteration}')


    def save_adjoint_output(self, iteration, error, convective_coefficient, T, output_folder):

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        file_name = f"{output_folder}/adjoint.csv"
        h_conv_file_name = f"{output_folder}/adjoint_h_conv.csv"
        T_file_name = f"{output_folder}/adjoint_T.csv"
        
        # If the file doesn't exist, create it with headers
        if iteration == 0 or not os.path.exists(file_name):
            with open(file_name, 'w', newline="") as f:
                f.write("iteration, error\n")

        else:
            with open(file_name, 'a') as f:
                writer = csv.writer(f, delimiter=",")
                formatted_data = [iteration, f"{error:.4f}"]
                writer.writerow(formatted_data)

        # If the file doesn't exist, create it with headers
        if iteration == 0 or not os.path.exists(h_conv_file_name):
            with open(h_conv_file_name, 'w') as f:
                f.write("")

        else:
            with open(h_conv_file_name, 'a') as f:
                writer = csv.writer(f)
                formatted_data = [f"{h:.4f}" for h in convective_coefficient]
                writer.writerow(formatted_data)

        # If the file doesn't exist, create it with headers
        if iteration == 0 or not os.path.exists(T_file_name):
            with open(T_file_name, 'w') as f:
                f.write("")

        else:
            with open(T_file_name, 'a') as f:
                writer = csv.writer(f)
                formatted_data = [f"{t:.4f}" for t in T]
                writer.writerow(formatted_data)


