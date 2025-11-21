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

# Import MPC controller with simulation model
# from source.simulation_model.finite_difference_3d import FiniteDifference3D
# from source.simulation_model.mpc_controller import MPCController
# from source.simulation_model.data_manager import DataManager
# from source.simulation_model.adjoint_optimizer import AdjointTransient

class ExperimentalMPCController:
    """Experimental MPC controller using a simulation model for prediction."""

    def __init__(self, n_region=1, n_mfc=9, dt_mpc= 60.0):
        """Initialize the experimental MPC controller.
        :param n_region: Number of temperature regions to control (default: 1 for full-plate control)
        :param n_mfc: Number of MFCs available for control (default: 9)
        """
        self.n_region = n_region
        self.n_mfc = n_mfc

        # # Initialize the simulation model
        # self.simulation_model = FiniteDifference3D()

        # # Initialize data manager for handling simulation data
        # self.data_manager = DataManager()

        # # Initialize the adjoint optimizer for MPC
        # self.adjoint_optimizer = AdjointTransient(self.simulation_model, self.data_manager)

        # Default parameters for MPC (can be updated by the UI)
        self.prediction_horizon = 3  # Number of future steps to predict
        self.control_horizon = 1  # Number of control steps to apply
        self.control_weight = 0.1  # Weight for control effort in cost function
        self.temperature_setpoint = 60.0  # Desired temperature setpoint in Celsius
        self.time_step = 60.0  # Time step in seconds

    def compute_mpc_control_action(self, current_temperatures, temperature_shape, current_flow_rates):
        """Compute the optimal MFC flow rates using MPC.
        :param current_temperatures: Current temperatures of the regions
        :param temperature_shape: Shape of the temperature array (H_sub, W_sub)
        :param current_flow_rates: Current flow rates of the MFCs
        :return: Optimal flow rates for the MFCs
        """
        # Update simulation model with current state
        self.simulation_model.set_initial_conditions(current_temperatures, current_flow_rates)

        # Run adjoint optimization to compute optimal control actions
        optimal_flow_rates = self.adjoint_optimizer.optimize(
            prediction_horizon=self.prediction_horizon,
            control_weight=self.control_weight,
            temperature_setpoint=self.temperature_setpoint,
            time_step=self.time_step
        )

        return optimal_flow_rates