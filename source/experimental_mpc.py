'''
Copyright 2024-2025, the Active Cooling Application Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import copy
import os
from source.simulation_model.finite_difference_3d import FiniteDifferenceSolver
from source.simulation_model.time_manager import TimeManager
from source.simulation_model.params import ParametersHandler
from scipy.optimize import minimize
from source.simulation_model.adjoint_optimizer import AdjointTransient
from source.simulation_model.data_manager import DataManager

class ExperimentalMPCController:
    """Experimental MPC controller using a simulation model for prediction."""

    def __init__(self, n_region=1, n_mfc=9, mpc_prediction_horizon=3, mpc_control_horizon=1, mpc_control_weight=0.1, verbose=False):
        """Initialize the experimental MPC controller.
        :param n_region: Number of temperature regions to control (default: 1 for full-plate control)
        :param n_mfc: Number of MFCs available for control (default: 9)
        """
        self.params = ParametersHandler(os.path.join(os.path.dirname(__file__), "experimental_mpc_params.txt"))
        self.n_region = n_region
        self.n_mfc = n_mfc
        self.verbose = verbose

        # Default parameters for MPC (can be updated by the UI)
        self.mpc_prediction_horizon = mpc_prediction_horizon  # Number of future steps to predict
        self.mpc_control_horizon = mpc_control_horizon  # Number of control steps to apply
        self.mpc_control_weight = mpc_control_weight  # Weight for control effort in cost function

        self.temperature_setpoint = 60.0  # Desired temperature setpoint in Celsius
        self.time_step = 60.0  # Time step in seconds

        self.system_model = None # system_model will be built before each MPC computation

    def _build_simulation_model(self):
        """Create a finite difference 3D simulation model of the cooling plate."""
        # 1) Create a time manager starting at 0
        time_manager = TimeManager(self.params)

        # 2) Create a copy of params 
        params_copy = copy.deepcopy(self.params)
        
        # 3) Create the finite difference solver model
        model = FiniteDifferenceSolver(params=params_copy, time_manager=time_manager)

        print("FD model grid:", model.params.nx, model.params.ny, model.params.nz)


        return model
    
    def _set_initial_temperature_from_camera(self, model, current_temperatures, temperature_shape):
        """
        Initialize the FD solver temperature field T(x,y,z) using 2D camera measurements.
        :param model: The simulation model
        :param measured_temperature_vector: flattened vector from subgrid
        :param temperature_shape: Shape of the temperature array (H_sub, W_sub)
        """
        # Reshape the flattened temperature vector to 2D
        cam_H, cam_W = temperature_shape
        temperature_2D = current_temperatures.reshape((cam_H, cam_W))

        nx, ny, nz  = model.params.nx, model.params.ny, model.params.nz

        # Resize to FD grid resolution (nx * ny)
        temp_sized = np.zeros((ny, nx), dtype=float)
        for i in range(nx):
            for j in range(ny):
                # Map (i,j) in FD grid to corresponding index in temperature_2D
                # using simple nearest-neighbor mapping 
                # TODO: use interpolation for better accuracy
                iy = int(j * cam_H / ny)
                ix = int(i * cam_W / nx)

                iy = min(iy, cam_H - 1)
                ix = min(ix, cam_W - 1)
                
                temp_sized[i, j] = temperature_2D[iy, ix]
        
        # Set the temperature field in the FD model (assuming uniform in z)
        temperature_3D = np.repeat(temp_sized[:, :, np.newaxis], nz, axis=2) # shape (nx, ny, nz)
        model.T = temperature_3D.reshape(-1, order="F")

        print("Mapped FD T range:", model.T.min(), model.T.max())

    def _apply_flow_to_boundary(self, model, flow_rates):
        """
        Apply the MFC flow rates to the model boundary conditions.
        :param model: The simulation model
        :param flow_rates: Flow rates for each MFC
        """
        Q = flow_rates/300 # Normalize flow rates to [0,1] assuming max flow rate is 300 L/min
        Q = np.clip(Q, -1, 1)  # Ensure flow rates are within valid range
        
        model.boundary.set_inlet_configuration(Q)
    
    def simulate_trajectory(self, model, Q_sequence, face_id, target_temperature):
        model = copy.deepcopy(model)
        T_tops = {}

        for step, Q in enumerate(Q_sequence):
            model.boundary.set_inlet_configuration(Q)
            model.time_manager.update_time()
            model.solve()

            T_face = model.get_temperature_face(face_id)
            T_tops[step] = T_face.copy()

        return T_tops
    
    def controller_cost_function(self, model, Q_sequence, face_id, target_temperature):
        T_tops = self.simulate_trajectory(model, Q_sequence, face_id, target_temperature)

        mse = 0.
        for T_face in T_tops.values():
            target_array = np.full_like(T_face, target_temperature)
            mse += np.mean((T_face - target_array)**2)

        mse /= len(T_tops)
        cost = mse

        for t in range(1, min(self.mpc_control_horizon, len(Q_sequence))):
            dQ = Q_sequence[t] - Q_sequence[t-1]
            cost += self.control_weight * np.sum(dQ**2)

        return cost


    def evaluate_cost(self, model, Q_sequence, face_id, target_temperature):
        return self.controller_cost_function(model, Q_sequence, face_id, target_temperature)
    
    def compute_gradient_over_horizon(self, model, Q_sequence, face_id, target_temperature, epsilon=0.1):
        """
        Compute the gradient of the cost function with respect to each Q_t,i using forward differences.
        Output shape: (mpc_prediction_horizon, 5)
        """
        N, D = Q_sequence.shape
        grad = np.zeros_like(Q_sequence)
        base_cost = self.evaluate_cost(model, Q_sequence, face_id, target_temperature)

        for t in range(N):
            if t >= self.mpc_control_horizon:
                # No gradient computation beyond control horizon
                continue
            for i in range(D):

                # Use central difference 
                Q_plus = Q_sequence.copy()
                Q_minus = Q_sequence.copy()

                Q_plus[t, i] += epsilon
                Q_minus[t, i] -= epsilon

                Q_plus[t, i] = np.clip(Q_plus[t, i], -1, 1)
                Q_minus[t, i] = np.clip(Q_minus[t, i], -1, 1)

                cost_plus = self.evaluate_cost(model, Q_plus, face_id, target_temperature)
                cost_minus = self.evaluate_cost(model, Q_minus, face_id, target_temperature)

                grad[t, i] = (cost_plus - cost_minus) / (2 * epsilon)


        return grad

    def compute_mpc_control_action(self, current_temperatures, temperature_shape, current_flow_rates):
        """Compute the optimal MFC flow rates using MPC.
        :param current_temperatures: Current temperatures of the regions
        :param temperature_shape: Shape of the temperature array (H_sub, W_sub)
        :param current_flow_rates: Current flow rates of the MFCs
        :return: Optimal flow rates for the MFCs
        """
        # 1) Build simulation model
        model = self._build_simulation_model()

        # 2) Set initial temperature condition from camera measurement
        self._set_initial_temperature_from_camera(model, current_temperatures, temperature_shape=temperature_shape)

        # 3) Apply current flow rates to the model boundary
        self._apply_flow_to_boundary(model, current_flow_rates)
        print("Boundary inlets:", model.boundary.inlet_configuration)

        # 4) Adjoint reconstruction of top boundary h(x,y) ###########################################

        # 4.1) Check if there is a heat load perturbation (if the plate temperature is above target), if so, set T_inf high for reconstruction
        target_temp = self.temperature_setpoint

        current_plate_temp = float(np.mean(model.get_temperature_face(5))) # Top face id=5
        if current_plate_temp > target_temp:
            new_T_inf = 250.0
        else:
            new_T_inf = current_plate_temp

        # 4.2) reset the top face boundary condition to a default convective coefficient (20) and T_inf
        model.boundary.reset_boundary(5, new_h=20.0, new_T_inf=new_T_inf)

        # 4.3) Get the two snapshots needed for adjoint reconstruction
        if not hasattr(self, 'previous_plate_temperature'):
            # First MPC iteration, no snapshot available yet
            self.previous_plate_temperature = model.get_temperature_face(face_id=5)

        previous_T = self.previous_plate_temperature
        current_T = model.get_temperature_face(face_id=5)

        # 4.3) run adjoint reconstruction and apply reconstructed h

        data_manager = DataManager(model.params, model.points)

        # Choose adjoint solver type
        adjoint = AdjointTransient(model.params, model, data_manager, target_snapshots=[previous_T, current_T])

        h_reconstructed = adjoint.run_nonlinear(return_h=True)

        # Update previous temperature face for next MPC iteration
        self.previous_plate_temperature = current_T.copy()

        # Ensure the shape of h_reconstructed matches the number of boundary points on the face
        if h_reconstructed.ndim == 2 and h_reconstructed.shape[0] == 1:
            h_reconstructed = h_reconstructed[0]

        #  apply the reconstructed coefficients directly
        model.boundary.apply_reconstructed_h(5, h_reconstructed)

        ###############################################################################

        # Use the controller's internal model for prediction
        predict_model = copy.deepcopy(model)

        # 4) initialize Q sequence
        N = self.mpc_prediction_horizon
        D = len(predict_model.boundary.inlet_configuration) # number of actuators
        
        if hasattr(self, 'previous_Q_sequence'):
            Q0_sequence = np.vstack([self.previous_Q_sequence[1:], self.previous_Q_sequence[-1]])
        else:
            current_Q = predict_model.boundary.inlet_configuration
            Q0_sequence = np.tile(current_Q, (N, 1))  # Initialize with the current Q repeated N times
            
        # Enforce move-blocking on the initial guess
        if self.mpc_control_horizon < N:
            Q0_sequence[self.mpc_control_horizon:] = Q0_sequence[self.mpc_control_horizon - 1]

        # Constraints parameters for outlet requirement
        beta   = 25.0   # sharpness for softmin (15–50 works well)
        margin = 1e-3    # require strictly < 0; set to e.g. 1e-3 for a safety margin

        Q0_flat = Q0_sequence.flatten()
        Q0 = Q0_flat.reshape(N, D)

        if not np.any(Q0[0] < -margin):
            j = int(np.argmin(Q0[0]))    # pick the smallest entry
            Q0[0, j] = -max(margin, 1e-3)
        Q0_flat = np.clip(Q0, -1, 1).flatten()

        face_id = 5  # top face
        target_temperature = self.temperature_setpoint

        # 5) define objective and gradient functions

        def objective(Q_flat):
            Q_seq = Q_flat.reshape(N, D)
            # enforce no arrangement and flow rate changes beyond control horizon
            if self.mpc_control_horizon < N:
                Q_seq[self.mpc_control_horizon:] = Q_seq[self.mpc_control_horizon - 1]
            # compute cost
            cost = self.evaluate_cost(predict_model, Q_seq, face_id, target_temperature)
            return cost

        def gradient(Q_flat):
            Q_seq = Q_flat.reshape(N, D)
            # enforce no arrangement and flow rate changes beyond control horizon
            if self.mpc_control_horizon < N:
                Q_seq[self.mpc_control_horizon:] = Q_seq[self.mpc_control_horizon - 1]
            # compute gradient
            grad_matrix = self.compute_gradient_over_horizon(predict_model, Q_seq, face_id, target_temperature)
            return grad_matrix.flatten()

        def callback(Q_flat):
            Q_seq = Q_flat.reshape(N, D)
            if self.mpc_control_horizon < N:
                Q_seq[self.mpc_control_horizon:] = Q_seq[self.mpc_control_horizon - 1]

            # 1) Evaluate the cost and store relevant information
            _ = self.evaluate_cost(predict_model, Q_seq, face_id, target_temperature)

            # 2) Simulate trajectory and store average predicted temperatures for the prediction horizon
            preds = self.simulate_trajectory(predict_model, Q_seq, face_id, target_temperature)
            # spatial average for the face for each prediction step
            Tpred_avgs = np.array([np.mean(Tk) for k, Tk in sorted(preds.items())], dtype=float)
            Tpred_full = [Tk.copy() for k, Tk in sorted(preds.items())]

        # 6) define constraints (at least one outlet per step)

        def _softmin(x, beta):
            m = x.min()
            return m - (1.0/beta) * np.log(np.sum(np.exp(-beta*(x - m))))

        def outlet_constraint_factory(step_idx):
            """
            Constraint for a single prediction step `step_idx`:
            min_j(Q[step_idx, j]) <= -margin  (at least one outlet at that step).
            Implemented via softmin.
            """
            def outlet_constraint_step(Q_flat):
                Q_seq = Q_flat.reshape(N, D)
                Q_step = Q_seq[step_idx]               # shape (D,)
                smin = _softmin(Q_step + margin, beta) # approx min_j(Q_step_j + margin)
                return -smin                           # >= 0 when min(Q_step)+margin <= 0
            return outlet_constraint_step

        def outlet_jacobian_factory(step_idx):
            """
            Jacobian for outlet constraint at step `step_idx`.
            Only the entries corresponding to that step (its D actuators) are non-zero.
            """
            def outlet_jac_step(Q_flat):
                Q_seq = Q_flat.reshape(N, D)
                Q_step = Q_seq[step_idx]               # shape (D,)

                x  = Q_step + margin
                xm = x.min()
                w  = np.exp(-beta * (x - xm))
                w /= np.sum(w)                         # weights over actuators at this step

                J = np.zeros(N * D, dtype=float)
                start = step_idx * D                   # block for this time step
                J[start:start + D] = w                 # d(-softmin)/dQ_step_j = w_j
                return J
            return outlet_jac_step

        bounds = [(-1, 1)] * (N * D)
        # One outlet constraint per free step (0..mpc_control_horizon-1)
        num_free_steps = min(self.mpc_control_horizon, N)
        constraints = []

        for s in range(num_free_steps):
            constraints.append({
                'type': 'ineq',
                'fun': outlet_constraint_factory(s),
                'jac': outlet_jacobian_factory(s)
            })

        # 7) run optimization

        result = minimize(
            fun=objective,
            x0=Q0_flat,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'maxiter': 50, 'ftol': 1e-6, 'disp': True},
            callback=callback
        )

        Q_opt_sequence = result.x.reshape(N, D)

        print("Q_opt_sequence:\n", Q_opt_sequence)
        print("First control action Q0:", Q_opt_sequence[0])

        if self.mpc_control_horizon < N:
            Q_opt_sequence[self.mpc_control_horizon:] = Q_opt_sequence[self.mpc_control_horizon - 1]
        self.predicted_temperature = self.simulate_trajectory(predict_model, Q_opt_sequence, face_id, target_temperature)

        self.previous_Q_sequence = Q_opt_sequence


        # 8) Return only the first control action (Q0), the final cost, and the predicted temperature trajectory
        return Q_opt_sequence[0], result.fun, self.predicted_temperature 





        # # Run adjoint optimization to compute optimal control actions
        # optimal_flow_rates = self.adjoint_optimizer.optimize(
        #     prediction_horizon=self.prediction_horizon,
        #     control_weight=self.control_weight,
        #     temperature_setpoint=self.temperature_setpoint,
        #     time_step=self.time_step
        # )

        #  # 2) Prepare Q-sequence (initial guess)
        # N = self.prediction_horizon
        # D = self.n_mfc

        # Q0 = np.zeros(D)
        # Q_sequence = np.tile(Q0, (N, 1))

        # # 3) Simple descent (placeholder)
        # best_cost = self.evaluate_cost(model, Q_sequence, face_id=0,
        #                             target_temperature=self.temperature_setpoint)

        # # 4) Return the first control action
        # return optimal_flow_rates 