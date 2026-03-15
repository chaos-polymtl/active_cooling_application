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
import time
from source.simulation_model.finite_difference_3d import FiniteDifferenceSolver
from source.simulation_model.time_manager import TimeManager
from source.simulation_model.params import ParametersHandler
from scipy.optimize import minimize
from source.simulation_model.adjoint_optimizer import AdjointTransient
from source.simulation_model.data_manager import DataManager
from source.simulation_model.utility import _copy_boundary_shared_surrogate

def _copy_boundary_shared_surrogate(boundary):
    """Shallow-copy boundary, sharing torch modules but copying numpy arrays."""
    import torch
    new_b = copy.copy(boundary)
    for attr_name in vars(boundary):
        attr = getattr(boundary, attr_name)
        if isinstance(attr, torch.nn.Module):
            setattr(new_b, attr_name, attr)
        elif isinstance(attr, np.ndarray):
            setattr(new_b, attr_name, attr.copy())
        else:
            try:
                setattr(new_b, attr_name, copy.deepcopy(attr))
            except Exception:
                setattr(new_b, attr_name, attr)
    return new_b

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

        self.temperature_setpoint = 30.0  # Desired temperature setpoint in Celsius
        self.time_step = 70.0  # Time step in seconds

        self.system_model = None # system_model will be built before each MPC computation

        self._pretrained_boundary = None
        self._preload_surrogate()

    def _preload_surrogate(self):
        """Load the PyTorch surrogate once on the main thread at startup."""
        import torch
        model = self._build_simulation_model()
        # Store the fully loaded boundary (with surrogate weights) as a template
        self._pretrained_boundary = model.boundary

    def _build_simulation_model(self):
        time_manager = TimeManager(self.params)
        params_copy = copy.deepcopy(self.params)
        model = FiniteDifferenceSolver(params=params_copy, time_manager=time_manager)

        if self._pretrained_boundary is not None:
            model.boundary = _copy_boundary_shared_surrogate(self._pretrained_boundary)

        model.T = np.ascontiguousarray(model.T, dtype=np.float64)
        if hasattr(model, 'points'):
            model.points = np.ascontiguousarray(model.points, dtype=np.float64)

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

    def _jet_centers_xy(self, model, D: int):
        """
        Return jet center locations in the (x,y) plane of the top face.
        Assumptions:
        - D==5  : 5x1 line along x, centered in y
        - D==9  : 3x3 grid over (x,y), row-major
        - else  : evenly spaced along x, centered in y
        """
        Lx, Ly, _ = model.params.plate_dimensions
        y_mid = 0.5 * Ly

        if D == 5:
            xs = np.linspace(0.0, Lx, 5)
            return np.column_stack([xs, np.full(5, y_mid)])

        if D == 9:
            xs = np.linspace(0.0, Lx, 3)
            ys = np.linspace(0.0, Ly, 3)
            centers = [(x, y) for y in ys for x in xs]
            return np.array(centers, dtype=float)

        xs = np.linspace(0.0, Lx, D)
        return np.column_stack([xs, np.full(D, y_mid)])

    def _initial_outlet_from_top_hotspot(
        self,
        current_model,
        top_face_id: int = 5,
        prev_Q_init=None,
        prev_hot_xy=None,
        move_threshold=0.1,
        hotspot_top_n: int = 25,
        hotspot_power: float = 1.0,
    ):
        """
        If hotspot moved (distance > move_threshold): re-init Q_init as all closed except farthest outlet.
        Else: keep previous Q_init.

        Hotspot position is computed as a center-of-mass (COM) of the top-N hottest nodes on the face.
        Returns: (Q_init, hot_xy_current)
        """
        T_face = current_model.get_temperature_face(top_face_id)
        face_nodes = current_model.boundary.dict_boundary_points[top_face_id]

        n = int(min(max(hotspot_top_n, 1), len(T_face)))
        top_idx = np.argpartition(T_face, -n)[-n:]
        top_global = np.asarray(face_nodes, dtype=int)[top_idx]

        pts = current_model.points[top_global]
        xy = pts[:, :2]

        T_top = np.asarray(T_face, dtype=float)[top_idx]
        w = (T_top - T_top.min())
        if hotspot_power != 1.0:
            w = np.power(w, hotspot_power)
        w = np.maximum(w, 1e-12)

        hot_xy = (w[:, None] * xy).sum(axis=0) / w.sum()

        moved = False
        if prev_hot_xy is not None:
            prev_hot_xy = np.asarray(prev_hot_xy, dtype=float)
            dist = float(np.linalg.norm(hot_xy - prev_hot_xy))
            moved = dist > move_threshold

        D = len(current_model.boundary.inlet_configuration)

        if (prev_Q_init is None) or moved:
            jet_xy = self._jet_centers_xy(current_model, D)
            d2 = (jet_xy[:, 0] - hot_xy[0]) ** 2 + (jet_xy[:, 1] - hot_xy[1]) ** 2
            j_far = int(np.argmax(d2))

            Q_init = np.zeros(D, dtype=float)
            Q_init[j_far] = -1.0
            return Q_init, hot_xy

        return np.array(prev_Q_init, dtype=float).copy(), hot_xy

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

        model.T = np.ascontiguousarray(model.T, dtype=np.float64)
        if hasattr(model, 'points'):
            model.points = np.ascontiguousarray(model.points, dtype=np.float64)

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

        mpc_control_horizon = min(self.mpc_control_horizon, len(Q_sequence))
        for t in range(1, mpc_control_horizon):
            dQ = Q_sequence[t] - Q_sequence[t-1]
            cost += self.control_weight * np.sum(dQ**2) / mpc_control_horizon

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
        start_time = time.time()
        print(f"[MPC] iteration start at t={start_time:.2f}")

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

        adjoint_params = copy.deepcopy(model.params)
        adjoint_model  = copy.deepcopy(model)
        adjoint_model.boundary = _copy_boundary_shared_surrogate(model.boundary)
        data_manager = DataManager(adjoint_params, adjoint_model.points)
        adjoint = AdjointTransient(adjoint_params, adjoint_model, data_manager, target_snapshots=[previous_T.copy(), current_T.copy()])

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

        predict_model.T = np.ascontiguousarray(predict_model.T, dtype=np.float64)
        if hasattr(predict_model, 'points'):
            predict_model.points = np.ascontiguousarray(predict_model.points, dtype=np.float64)

        # 4) initialize Q sequence
        N = self.mpc_prediction_horizon
        D = len(predict_model.boundary.inlet_configuration) # number of actuators
        
        # Hotspot-based initialization with memory (same logic as simulation MPC)
        prev_Q_init = getattr(self, "prev_Q_init", None)
        prev_hot_xy = getattr(self, "prev_hot_xy", None)

        Q_init, hot_xy = self._initial_outlet_from_top_hotspot(
            model,
            top_face_id=5,
            prev_Q_init=prev_Q_init,
            prev_hot_xy=prev_hot_xy,
            move_threshold=getattr(model.params, "mpc_hotspot_move_threshold", 0.1),
            hotspot_top_n=getattr(model.params, "mpc_hotspot_top_n", 25),
            hotspot_power=getattr(model.params, "mpc_hotspot_power", 1.0),
        )

        self.prev_Q_init = Q_init.copy()
        self.prev_hot_xy = hot_xy.copy()

        Q0_sequence = np.tile(Q_init, (N, 1))
            
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
            if getattr(model.params, "apply_gradient_mask", False):
                keep_jet = getattr(model.params, "gradient_keep_jets", None)
                if keep_jet is not None:
                    mask = np.zeros_like(grad_matrix)
                    mask[:, keep_jet] = 1.0
                    grad_matrix *= mask
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
            options={'maxiter': 50, 'ftol': 1e-2, 'disp': True},
            callback=callback
        )

        Q_opt_sequence = result.x.reshape(N, D)

        print("Q_opt_sequence:\n", Q_opt_sequence)
        print("First control action Q0:", Q_opt_sequence[0])

        if self.mpc_control_horizon < N:
            Q_opt_sequence[self.mpc_control_horizon:] = Q_opt_sequence[self.mpc_control_horizon - 1]
        self.predicted_temperature = self.simulate_trajectory(predict_model, Q_opt_sequence, face_id, target_temperature)

        self.previous_Q_sequence = Q_opt_sequence

        end_time = time.time()
        print(f"[MPC] iteration end at t={end_time:.2f}, duration: {end_time - start_time:.2f} seconds")

        # 8) Return only the first control action (Q0), the final cost, and the predicted temperature trajectory
        return Q_opt_sequence[0], result.fun, self.predicted_temperature 