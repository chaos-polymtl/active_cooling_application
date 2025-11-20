import numpy as np
import copy
from scipy.optimize import minimize
import os, csv

class MPCController:
    def __init__(self, system_model, mpc_prediction_horizon=10, mpc_control_horizon=3, informed=True, verbose = False):
        """
        Initialize the MPC controller with a system model, prediction horizon, and control horizon.
        :param system_model: The system model to control, which should have a method to solve the PDE.
        :param mpc_prediction_horizon: The number of time steps to predict into the future.
        :param informed: whether to use the exact system model with outside loop parameter
        if informed=False, the interal copy of the system model resets its top boundary condition
        to an uninformed state before running the heat load reconstruction.
        """
        self.system_model = copy.deepcopy(system_model)
        self.mpc_prediction_horizon = mpc_prediction_horizon
        self.mpc_control_horizon = mpc_control_horizon
        self.informed = informed
        self.verbose = verbose

        self.iter_costs = []
        self.grad_norm_history = []
        self.iter_mse = []
        self.iter_rmse = []
        self._last_grad_norm = None
        self._last_mse = None
        self._last_rmse = None
        self._last_total_cost = None

    def log(self, message):
        if self.verbose:
            print(message)

    def controller_cost_function(self, model, Q_sequence, face_id, target_temperature, control_weight=0.01):
        """
        Compute the spatial cost over all time steps in the prediction horizon.
        Include a penalty on changes in control actions.

        Currently: Returns the normalized MSE across space and prediction steps
        """
        T_tops = self.simulate_trajectory(model, Q_sequence, face_id, target_temperature)

        mse = 0.0
        cost = 0.0

        for T_face in T_tops.values():
            # If target_temperature is scalar, broadcast it
            if np.isscalar(target_temperature):
                target_array = np.full_like(T_face, target_temperature)
            else:
                target_array = target_temperature

            # spatial MSE (mean squared error over face) 
            mse += np.mean((T_face - target_array) ** 2)
        
        # average over prediction steps
        mse /= len(T_tops)

        # saving these for logging
        self._last_mse = float(mse)
        self._last_rmse = float(np.sqrt(mse))

        cost = mse

        # Control effort cost

        mpc_control_horizon = min(self.mpc_control_horizon, len(Q_sequence))

        for t in range(1, mpc_control_horizon):
            dQ = Q_sequence[t] - Q_sequence[t - 1]
            cost += control_weight * np.sum(dQ ** 2) / mpc_control_horizon # normalize by control horizon length

        return cost

    def evaluate_cost(self, model, Q_sequence, face_id, target_temperature):
        base_mse = self.controller_cost_function(model, Q_sequence, face_id, target_temperature)
        total = base_mse # + self._outlet_penalty(Q_sequence)
        self._last_total_cost = float(total)
        return total

    def simulate_trajectory(self, model, Q_sequence, face_id, target_temperature):
        """
        Simulates the system forward using a sequence of Qs.
        Returns a list of face temperatures at each time step (not just the final one).
        """
        model = copy.deepcopy(model)

        t0 = model.time_manager.current_time
        s0 = model.time_manager.current_step

        T_tops = {}  # dict: step -> temperature vector on face

        for step, Q in enumerate(Q_sequence):
            model.boundary.set_inlet_configuration(Q)
            model.time_manager.update_time()
            model.solve()

            T_face = model.get_temperature_face(face_id)
            T_tops[step] = T_face.copy()
            # self.log(f"[Sim] Step {step} | T_avg: {np.mean(T_face):.2f} | T_target: {target_temperature}")

        model.time_manager.current_time = t0
        model.time_manager.current_step = s0

        return T_tops

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

    def compute_mpc_control_action(self, current_model, face_id, target_temperature, informed=True):
        """
        Run the MPC optimization to compute the optimal control action Q0.
        Optimize over the prediction horizon using scipy.optimize.minimize with SLSQP.
        Steps for the uninformed case:
        1) Check if there is a heat load perturbation (if the plate temperature is above target)
        2) If the plate temperature is above target, reset the top face boundary condition to a default convective coefficient (20) and T_inf,
            where T_inf is set to 250C if the plate temperature is above target, or to the current plate temperature otherwise.
        3) Run the adjoint reconstruction to get the h values for the top face (recreate the heat load conditions)
        4) Initialize Q sequence (from previous solution or current model state)
        5) Define objective and gradient functions
        6) Define constraints (at least one outlet per step)
        7) Run optimization
        8) Return optimal Q0 and predicted temperature trajectory.
        """

        if informed is not None:
            self.informed = informed

        # --- Uninformed case: reset boundary and apply reconstructed h (if reconstruction is enabled) ---
        if not self.informed:
            face_id = getattr(self.system_model.params, "target_face_id", 5)
            self.log(f"[MPC] Uninformed internal model → resetting boundary condition on face {face_id}.")

            if getattr(self.system_model.params, "mpc_reconstruction", True):
                self.log("[MPC] Uninformed mode with reconstruction enabled.")
        
                # 1) Check if there is a heat load perturbation (if the plate temperature is above target), if so, set T_inf high for reconstruction
                target_temp = getattr(self.system_model.params, "target_temperature", 60.0)
                # Handle both scalar and list/array cases
                if isinstance(target_temp, (list, tuple, np.ndarray)):
                    target_temp = float(target_temp[0])
                else:
                    target_temp = float(target_temp)

                current_plate_temp = float(np.mean(self.system_model.get_temperature_face(face_id)))
                if current_plate_temp > target_temp:
                    new_T_inf = 250.0
                    self.log(f"[MPC] Current plate temperature {current_plate_temp:.2f}°C exceeds target {target_temp:.2f}°C. Setting T_inf to {new_T_inf}°C for reconstruction.")
                else:
                    new_T_inf = current_plate_temp

                # 2) reset the top face boundary condition to a default convective coefficient (20) and T_inf
                self.system_model.boundary.reset_boundary(face_id, new_h=20.0, new_T_inf=new_T_inf)

                # define the previous and current top face temperatures for adjoint reconstruction
                if not hasattr(self, 'previous_plate_temperature'):
                    self.log("[MPC] Initializing previous temperature face for first iteration.")
                    self.previous_plate_temperature = current_model.get_temperature_face(face_id=5)
                previous_T = self.previous_plate_temperature
                current_T = current_model.get_temperature_face(face_id=face_id)

                # 3) run adjoint reconstruction and apply reconstructed h
                self.log("[MPC] Running adjoint reconstruction (return_h=True)...")

                from src.adjoint_optimizer import AdjointTransient
                from src.data_manager import DataManager

                data_manager = DataManager(self.system_model.params, self.system_model.points)

                # Choose adjoint solver type
                adjoint = AdjointTransient(self.system_model.params, self.system_model, data_manager, target_snapshots=[previous_T, current_T])
                h_reconstructed = adjoint.run_nonlinear(return_h=True)

                # Update previous temperature face for next MPC iteration
                self.previous_plate_temperature = current_T.copy()

                # Ensure the shape of h_reconstructed matches the number of boundary points on the face
                if h_reconstructed.ndim == 2 and h_reconstructed.shape[0] == 1:
                    h_reconstructed = h_reconstructed[0]

                #  apply the reconstructed coefficients directly
                self.system_model.boundary.apply_reconstructed_h(face_id, h_reconstructed)
                self.log("[MPC] Applied reconstructed h coefficients to MPC model boundary.")

            else:
                self.log("[MPC] Uninformed mode without reconstruction: using boundary condition before perturbation.")
                self.system_model.boundary.reset_boundary(face_id, new_h=80.0, new_T_inf=40.0)

        # Use the controller's internal model for prediction
        predict_model = copy.deepcopy(self.system_model)

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

        self.iter_costs = []
        self.grad_norm_history = []
        self.iter_mse = []
        self.iter_rmse = []
        self._last_grad_norm = None
        self._last_mse = None
        self._last_rmse = None
        self._last_total_cost = None
        self.iter_Q_sequences = [] # list of (N, 5) 
        self.iter_Tpred_avgs = []  # list of (N,) average predicted temperatures
        self.iter_Tpred_full = []  # list of (N, num_faces) full predicted temperatures

        # 5) define objective and gradient functions

        def objective(Q_flat):
            Q_seq = Q_flat.reshape(N, D)
            # enforce no arrangement and flow rate changes beyond control horizon
            if self.mpc_control_horizon < N:
                Q_seq[self.mpc_control_horizon:] = Q_seq[self.mpc_control_horizon - 1]
            # compute cost
            cost = self.evaluate_cost(predict_model, Q_seq, face_id, target_temperature)
            self.log(f"[Objective] MSE: {self._last_mse:.6g} | Q0: {Q_seq[0]}")
            return cost

        def gradient(Q_flat):
            Q_seq = Q_flat.reshape(N, D)
            # enforce no arrangement and flow rate changes beyond control horizon
            if self.mpc_control_horizon < N:
                Q_seq[self.mpc_control_horizon:] = Q_seq[self.mpc_control_horizon - 1]
            # compute gradient
            grad_matrix = self.compute_gradient_over_horizon(predict_model, Q_seq, face_id, target_temperature)
            self._last_grad_norm = float(np.linalg.norm(grad_matrix))
            self.log(f"[Gradient] Norm: {self._last_grad_norm:.2f}")
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

            # 3) Update iteration history
            self.iter_costs.append(self._last_total_cost)
            self.iter_mse.append(self._last_mse)
            self.iter_rmse.append(self._last_rmse)
            self.iter_Q_sequences.append(Q_seq.copy())
            self.iter_Tpred_avgs.append(Tpred_avgs.copy())
            self.iter_Tpred_full.append(Tpred_full)

            if self._last_grad_norm is None:
                gmat = self.compute_gradient_over_horizon(predict_model, Q_seq, face_id, target_temperature)
                self._last_grad_norm = float(np.linalg.norm(gmat))
            self.grad_norm_history.append(self._last_grad_norm)
            self.log(f"[Iter {len(self.iter_costs)}] Cost: {self.iter_costs[-1]:.2f} | ‖grad‖: {self._last_grad_norm:.2f}")

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
        if self.mpc_control_horizon < N:
            Q_opt_sequence[self.mpc_control_horizon:] = Q_opt_sequence[self.mpc_control_horizon - 1]
        self.predicted_temperature = self.simulate_trajectory(predict_model, Q_opt_sequence, face_id, target_temperature)

        self.previous_Q_sequence = Q_opt_sequence

        self.log(f"[MPC] Final cost: {result.fun:.2f}")
        self.log(f"[MPC] Optimized Q0: {Q_opt_sequence[0]}")

        try:
             # Infos de timing
            dt = float(getattr(current_model.params, "time_step", 1.0))
            step_idx = int(getattr(current_model.time_manager, "current_step", 0))
            out_dir = getattr(current_model.params, "output_path", None) or "."
            os.makedirs(out_dir, exist_ok=True)

            csv_path = os.path.join(out_dir, f"mpc_iter_traces_step_{step_idx}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                
                # En-tête
                header = ["iteration", "pred_step", "rel_time"]
                header += [f"Q{j}" for j in range(D)]
                header += ["T_pred_avg", "T_pred_full", "total_cost", "grad_norm"]
                # header = ["iteration", "pred_step", "rel_time",
                #         "Q0","Q1","Q2","Q3","Q4",
                #         "T_pred_avg", "T_pred_full", "total_cost", "grad_norm"]
                writer.writerow(header)

                n_iters = len(self.iter_Q_sequences)
                for it in range(n_iters):
                    Qseq = self.iter_Q_sequences[it]          # (N,D)
                    Tseq = self.iter_Tpred_avgs[it]           # (N,)
                    Tseq_full = self.iter_Tpred_full[it]      # list of length N
                    cost_it = float(self.iter_costs[it]) if it < len(self.iter_costs) else float("nan")
                    gnorm_it = float(self.grad_norm_history[it]) if it < len(self.grad_norm_history) else float("nan")

                    # une ligne par pas de prédiction k
                    for k in range(Qseq.shape[0]):
                        rel_time = (k+1) * dt  # temps relatif dans l’horizon

                        # Températures spatiales de la face (au lieu de juste la moyenne)
                        T_face_full = Tseq_full[k] 
                        T_face_serialized = ";".join(f"{float(val):.6g}" for val in T_face_full)

                        row = [it+1, k+1, rel_time,
                            *[float(x) for x in Qseq[k, :]],
                            float(Tseq[k]),
                            T_face_serialized,
                            cost_it, gnorm_it]
                        writer.writerow(row)

            self.log(f"[MPC] Wrote per-iteration traces to: {csv_path}")
        except Exception as e:
            self.log(f"[MPC] WARNING: could not write per-iteration CSV: {e}")

        # 8) Return only the first control action (Q0), the final cost, and the predicted temperature trajectory
        return Q_opt_sequence[0], result.fun, self.predicted_temperature 