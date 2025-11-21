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
from source.simulation_model import params
from source.simulation_model.finite_difference_3d import FiniteDifferenceSolver
from source.simulation_model.time_manager import TimeManager
from source.simulation_model.params import Params

# Import MPC controller with simulation model
# from source.simulation_model.finite_difference_3d import FiniteDifference3D
# from source.simulation_model.mpc_controller import MPCController
# from source.simulation_model.data_manager import DataManager
# from source.simulation_model.adjoint_optimizer import AdjointTransient

class ExperimentalMPCController:
    """Experimental MPC controller using a simulation model for prediction."""

    def __init__(self, n_region=1, n_mfc=9, verbose=False):
        """Initialize the experimental MPC controller.
        :param n_region: Number of temperature regions to control (default: 1 for full-plate control)
        :param n_mfc: Number of MFCs available for control (default: 9)
        """
        self.params = params
        self.n_region = n_region
        self.n_mfc = n_mfc
        self.verbose = verbose

        # Default parameters for MPC (can be updated by the UI)
        self.prediction_horizon = 3  # Number of future steps to predict
        self.control_horizon = 1  # Number of control steps to apply
        self.control_weight = 0.1  # Weight for control effort in cost function
        self.temperature_setpoint = 60.0  # Desired temperature setpoint in Celsius
        self.time_step = 60.0  # Time step in seconds

        self.system_model = None # system_model will be built before each MPC computation

    def _build_simulation_model(self):
        """Create a finite difference 3D simulation model of the cooling plate."""
        # 1) Create a time manager starting at 0
        time_manager = TimeManager(start_time=0.0, time_step=self.time_step)

        # 2) Create a copy of params 
        params_copy = copy.deepcopy(self.params)
        
        # 3) Create the finite difference solver model
        model = FiniteDifferenceSolver(params=params_copy, time_manager=time_manager)

        return model
    
    def _set_initial_temperature(self, model, measured_temperature_vector):
        """
        measured_temperature_vector: flattened vector from subgrid
        Must be mapped to model.T which is of size nx*ny*nz
        """
        # For now, we simply project the region-average onto the entire top surface
        # TODO: map the camera grid to FD mesh using interpolation
        T0 = measured_temperature_vector.mean()

        model.T[:] = T0

    def _apply_flow_to_boundary(self, model, Q):
        """
        Apply the MFC flow rates to the model boundary conditions.
        :param model: The simulation model
        :param Q: Flow rates for each MFC
        """
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

        for t in range(1, min(self.control_horizon, len(Q_sequence))):
            dQ = Q_sequence[t] - Q_sequence[t-1]
            cost += self.control_weight * np.sum(dQ**2)

        return cost


    def evaluate_cost(self, model, Q_sequence, face_id, target_temperature):
        return self.controller_cost_function(model, Q_sequence, face_id, target_temperature)


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
        self._set_initial_temperature(model, current_temperatures)

        # 3) Apply current flow rates to the model boundary
        self._apply_flow_to_boundary(model, current_flow_rates)

        # 2) Initialize adjoint optimizer with the simulation model

        # Update simulation model with current state
        self.simulation_model.set_initial_conditions(current_temperatures, current_flow_rates)

        # Run adjoint optimization to compute optimal control actions
        optimal_flow_rates = self.adjoint_optimizer.optimize(
            prediction_horizon=self.prediction_horizon,
            control_weight=self.control_weight,
            temperature_setpoint=self.temperature_setpoint,
            time_step=self.time_step
        )

         # 2) Prepare Q-sequence (initial guess)
        N = self.prediction_horizon
        D = self.n_mfc

        Q0 = np.zeros(D)
        Q_sequence = np.tile(Q0, (N, 1))

        # 3) Simple descent (placeholder)
        best_cost = self.evaluate_cost(model, Q_sequence, face_id=0,
                                    target_temperature=self.temperature_setpoint)

        # 4) Return the first control action
        return optimal_flow_rates 