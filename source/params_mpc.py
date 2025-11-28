'''
Copyright 2024-2025, the Active Cooling Application Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

class MPCParams:
    """Class to hold MPC parameters."""
    
    def __init__(self):
        # Time parameters
        self.start_time = 0.0  # Initial simulation time
        self.time_step = 60.0  # Time step size in seconds
        self.final_time = None  # Optional final simulation time

        # Grid resolution
        self.nx = 10  # Number of grid points in x-direction
        self.ny = 10  # Number of grid points in y-direction
        self.nz = 10  # Number of grid points in z-direction

        # Plate dimensions (in meters)
        self.plate_dimensions = (0.3, 0.3, 0.005)  # (length_x, length_y, thickness_z)

        # Material properties
        self.thermal_conductivity = 45
        self.density = 7850
        self.heat_capacity = 420

        # Initial temperature
        self.initial_condition = 25.0  # in Celsius
        # Note: params.initial_condiiton is only required to construct the FD solver. 
        # The actual initial condition for MPC predictions will be set from camera measurements.
        # Using _set_initial_temperature_from_camera method in ExperimentalMPCController.

        ######################## all other parameters ###########################

        # Boundary conditions
        self.robin_boundaries = [5]
        self.BC_types = ("no_flux", "no_flux", "no_flux", "no_flux", "surrogate", "robin")  # BC types for each face
        self.initial_convective_coefficient = {5:"80"}  # W/m2K
        self.T_inf = {5:"80"}  # Ambient temperatures for BCs

        # Surrogate
        self.nn_3x3_surrogate = True  # Use 3x3 MFC surrogate model
        self.surrogate_path = "source/simulation_model/3x3_surrogate.pth"  # Path to surrogate model file
        self.scale_factor = 1 # Use default scaling

        self.initial_inlet_configuration = (0, 0, 0, 0, 0, 0, 0, 0, -1)  # Initial MFC states (9 MFCs)
        self.jet_diameter = 0.0127
        self.fluid_conductivity = 0.026

        # adjoint
        self.adjoint_target_face = 5  # optimize the top face
        self.adjoint_optimize_face = 5  # optimize the top face
        self.adjoint_tolerance = 0.5  # tolerance for adjoint optimization
        self.adjoint_max_iterations = 50 # max iterations for adjoint optimization

        ######## other parameters that I need to add or else it gives errors ##########
        self.zone = None  # Dummy value
        self.zone_corners = None  # Dummy value
        self.idw_surrogate = False  # Dummy value
        self.nn_surrogate = False  # Dummy value
        self.nn_5x1_surrogate = None  # Dummy value

        self.dirichlet_boundaries = []  # Dummy value
        self.dirichlet_values = {}  # Dummy value
        self.neumann_boundaries = []  # Dummy value
        self.neumann_values = {}  # Dummy value

        self.robin_natural_boundaries = []  # Dummy value
        self.robin_natural_values = {}  # Dummy value

        self.output_path = None          # no VTU output needed during experiments
        self.source = False  # Dummy value