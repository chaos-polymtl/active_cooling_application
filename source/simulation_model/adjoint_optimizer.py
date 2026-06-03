import os
import numpy as np
import csv
from scipy.ndimage import gaussian_filter

import copy
from scipy.linalg import solve
from source.simulation_model.finite_difference_3d import FiniteDifferenceSolverSS
from source.simulation_model.finite_difference_3d import FiniteDifferenceSolver
from source.simulation_model.finite_difference_3d import AdjointSolver
from scipy.optimize import minimize

from source.simulation_model.time_manager import TimeManager

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

    def run_nonlinear(self, return_h=False, initial_h=None):
        """
        Run the nonlinear adjoint optimization loop.

        :param return_h: Return the optimized h array instead of saving to file.
        :param initial_h: Warm-start h (from previous MPC step). If None, reads from boundary.
        """

        # Handle missing DataManager (MPC integration: no VTU output needed)
        if self.data_manager is None:
            def _noop_save_vtu(*args, **kwargs):
                pass
            self.data_manager = type("DummyDM", (), {"save_vtu": staticmethod(_noop_save_vtu)})()

        # Determine number of time steps
        if hasattr(self, 'target_T') and len(self.target_T) == 1 and hasattr(self, 'initial_temp'):
            n_time_steps = 1
        else:
            n_time_steps = int((self.params.final_time - self.params.start_time) / self.params.time_step)

        error = 1e6
        tolerance = self.params.adjoint_tolerance
        iteration = 0
        max_iteration = self.params.adjoint_max_iterations

        target_face = self.params.adjoint_target_face
        face_to_optimize = self.params.adjoint_optimize_face

        nx, ny, nz = self.params.nx, self.params.ny, self.params.nz
        n_points = nx * ny
        n_total = nx * ny * nz

        direct_temperature_solutions = np.zeros((n_time_steps, n_points))
        adjoint_temperature_solutions = np.zeros((n_time_steps, n_points))
        h_new = np.zeros((n_time_steps, n_points))

        # Adam optimizer state (replaces fixed-step gradient descent)
        moment   = np.zeros((n_time_steps, n_points))
        moment_2 = np.zeros((n_time_steps, n_points))
        beta_1   = 0.9
        beta_2   = 0.999
        adam_lr  = 0.1  # learning rate (was 0.0001 with plain gradient descent)

        h_coeff_to_output   = np.zeros(n_points * nz)
        temp_error_to_output = np.zeros(n_points * nz)

        # Initialise h values (warm-start or default from boundary)
        for i in range(n_time_steps):
            if initial_h is not None:
                h_new[i] = np.asarray(initial_h, dtype=float)
            else:
                h_new[i] = self.finite_difference.boundary.get_convective_coefficient_at_face(face_to_optimize)

        # ------------------------------------------------------------------
        # Create the adjoint solver ONCE outside the loop to avoid reloading
        # the NN surrogate model on every iteration (was a major bottleneck).
        # ------------------------------------------------------------------
        time_manager_adj = TimeManager(self.params)
        self.adjoint = AdjointSolver(self.params, time_manager_adj, self.target_T)

        # Read the actual T_inf from the direct solver's boundary.
        # The MPC calls reset_boundary() before running the adjoint, so T_inf
        # may differ from the params file value (e.g. 250 °C for heat-load mode).
        try:
            _default_T_inf = float(eval(str(self.params.T_inf[target_face])))
        except Exception:
            _default_T_inf = 25.0
        T_inf_recon = np.array([
            self.finite_difference.boundary.T_inf.get(int(idx), _default_T_inf)
            for idx in self.finite_difference.boundary.dict_boundary_points[target_face]
        ])

        # Build a proper 3D initial condition from the measured 2D face temperature.
        # Tiling extends the face values uniformly through the z-direction (thin-plate
        # assumption). This is much better than using a single scalar average.
        if hasattr(self, "initial_temp"):
            # initial_temp has shape (nx*ny,) matching face ordering (x fast, y slow)
            T_init_3d = np.tile(np.asarray(self.initial_temp, dtype=float), nz)
        else:
            T_init_3d = np.full(n_total, float(np.average(self.params.adjoint_target_array[0])))

        # ------------------------------------------------------------------
        # Main adjoint iteration loop
        # ------------------------------------------------------------------
        while error > tolerance and iteration < max_iteration:

            # ---- Reset the direct FD solver (reuse existing object) ----
            # Only reset mutable state: T field and time manager.
            # The boundary (including NN surrogate) is already configured.
            self.finite_difference.time_manager.current_time = self.params.start_time
            self.finite_difference.time_manager.current_step = 0
            time_step = 0
            self.finite_difference.T = T_init_3d.copy()

            # ---- FORWARD PASS ----
            while not self.finite_difference.time_manager.is_finished() and time_step < n_time_steps:
                idx = max(0, min(time_step, n_time_steps - 1))

                self.finite_difference.boundary.set_convective_heat_transfer_map_for_face(face_to_optimize, h_new[idx])

                id_h_coefficient_imposed = self.finite_difference.boundary.convective_coefficient.keys()
                for id in id_h_coefficient_imposed:
                    h_coeff_to_output[id] = self.finite_difference.boundary.convective_coefficient[id]

                if time_step != 0:
                    temp_error_on_face = self.finite_difference.get_temperature_face(target_face) - self.target_T[time_step - 1]
                else:
                    temp_error_on_face = self.finite_difference.get_temperature_face(target_face) * 0.0

                for i, index in enumerate(self.finite_difference.boundary.dict_boundary_points[target_face]):
                    temp_error_to_output[index] = temp_error_on_face[i]

                self.data_manager.save_vtu(self.finite_difference.time_manager.current_step, self.finite_difference.T,
                                           self.finite_difference.heat_flux, h_coeff_to_output, temp_error_to_output)

                self.finite_difference.solve()
                direct_temperature_solutions[time_step] = self.finite_difference.get_temperature_face(target_face)

                self.finite_difference.time_manager.update_time()
                time_step += 1

            # Final forward-pass error snapshot
            if time_step != 0:
                temp_error_on_face = self.finite_difference.get_temperature_face(target_face) - self.target_T[time_step - 1]
            else:
                temp_error_on_face = self.finite_difference.get_temperature_face(target_face) * 0.0

            for i, index in enumerate(self.finite_difference.boundary.dict_boundary_points[target_face]):
                temp_error_to_output[index] = temp_error_on_face[i]
            self.data_manager.save_vtu(self.finite_difference.time_manager.current_step, self.finite_difference.T,
                                       self.finite_difference.heat_flux, h_coeff_to_output, temp_error_to_output)

            # ---- BACKWARD (ADJOINT) PASS ----
            # Reset adjoint solver state without recreating it (avoids NN reload).
            time_manager_adj.current_time = self.params.start_time
            time_manager_adj.current_step = 0
            self.adjoint.lambda_t = np.zeros(n_total)

            while not time_manager_adj.is_finished() and time_step > 0:
                idx = max(0, min(time_step - 1, n_time_steps - 1))

                self.adjoint.boundary.set_convective_heat_transfer_map_for_face(face_to_optimize, h_new[idx])

                id_h_coefficient_imposed = self.adjoint.boundary.convective_coefficient.keys()
                for id in id_h_coefficient_imposed:
                    h_coeff_to_output[id] = self.adjoint.boundary.convective_coefficient[id]

                self.data_manager.save_vtu(time_step, self.adjoint.lambda_t, self.finite_difference.heat_flux,
                                           h_coeff_to_output, file_name_prefix="adjoint_output")

                self.adjoint.solve(direct_temperature_solutions[time_step - 1], self.target_T[time_step - 1])
                adjoint_temperature_solutions[time_step - 1] = self.adjoint.get_lambda_at_face(target_face)

                time_manager_adj.update_time()
                time_step -= 1

            self.data_manager.save_vtu(time_step, self.adjoint.lambda_t, self.finite_difference.heat_flux,
                                       h_coeff_to_output, file_name_prefix="adjoint_output")

            # ---- GRADIENT ----
            # Use the actual T_inf from the direct solver boundary (not the stale params value).
            gradient = (-1.0 / self.params.thermal_conductivity
                        * adjoint_temperature_solutions
                        * (direct_temperature_solutions - T_inf_recon[np.newaxis, :]))

            # ---- ADAM UPDATE (replaces fixed-step gradient descent) ----
            iteration_1based = iteration + 1
            moment   = beta_1 * moment   + (1.0 - beta_1) * gradient
            moment_2 = beta_2 * moment_2 + (1.0 - beta_2) * np.square(gradient)
            m_hat = moment   / (1.0 - beta_1 ** iteration_1based)
            v_hat = moment_2 / (1.0 - beta_2 ** iteration_1based)
            h_new = h_new - adam_lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            h_new = np.maximum(h_new, 0.0)

            # ---- CONVERGENCE CHECK ----
            error = (np.linalg.norm(direct_temperature_solutions - self.target_T, 2)
                     / np.sqrt(len(self.target_T) * len(self.target_T[0])))

            iteration += 1

        if return_h:
            return np.asarray(h_new, dtype=float), error, iteration

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
            # print(f'Iteration: {iteration}, Error: {error:.4f}')

            # Save adjoint output
            if self.params.adjoint_output_results:
                self.save_adjoint_output(iteration, error, new_h, current_T, "adjoint_output")

            beta = 0.9
            beta_2 = 0.999

            iteration += 1

        # print(f'Iterations: {iteration}')


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


