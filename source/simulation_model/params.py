import numpy as np

class ParametersHandler:
    def __init__(self, file_path):
        
        # Initialize default parameters
        self.steady = False
        self.save_output = False
        self.output_path = "output"
        self.start_time = 0
        self.final_time = 1
        self.time_step = 0.01
        self.thermal_conductivity = 1
        self.density = 1
        self.heat_capacity = 1
        self.plate_dimensions = [1, 1, 1]
        self.nx = 10
        self.ny = 10
        self.nz = 10
        self.initial_condition = 0
        self.BC_types  = ["no_flux", "no_flux", "no_flux", "no_flux", "dirichlet", "dirichlet"]
        n_faces = 6
        self.BC_T = np.zeros(n_faces)
        self.BC_flux = np.zeros(n_faces)
        self.initial_convective_coefficient = np.zeros(n_faces)
        self.T_inf = np.zeros(n_faces)
        self.source_term = np.zeros((self.nx, self.ny, self.nz))
        self.zone = False
        
        self.raw_parameters = {}

        # Control parameters
        self.apply_pid_control = False
        self.apply_mpc_control = False
        self.mpc_prediction_horizon = 1
        self.mpc_control_horizon = 1
        self.mpc_informed = True

        # Surrogate parameters
        self.idw_surrogate = False
        self.nn_surrogate = False
        self.nn_5x1_surrogate = False
        self.nn_3x3_surrogate = False   
        self.initial_inlet_configuration = None #version for the 5x1: np.zeros(5, dtype=float)

        # Adjoint parameters
        self.solve_adjoint = False
        self.adjoint_tolerance = 1 # was 0.5
        self.adjoint_max_iterations = 50
        self.adjoint_target_face = 0
        self.adjoint_target_T = 0
        self.adjoint_optimize_face = 0
        self.adjoint_step_size = 1
        self.adjoint_track_error = False
        self.adjoint_output_results = False

        # Perturbation parameters
        self.perturbation_type = None 
        self.perturbation_time = None
        self.perturbation_face_id = None
        # Gaussian Neumann perturbation (heat flux)
        self.gaussian_neumann_amplitude = None
        self.gaussian_neumann_center = None
        self.gaussian_neumann_sigma = None
        # Gaussian Robin perturbation (convective heat transfer coefficient map)
        self.gaussian_robin_h_amplitude = None
        self.gaussian_robin_center = None
        self.gaussian_robin_sigma = None
        self.gaussian_robin_T_inf = None

        # Echelon test parameters
        self.apply_echelon = False
        self.echelon_start_time = None
        self.echelon_jet_id = None
        self.echelon_value = None

        # Read parameters from file
        # --------------------------------------
        with open(file_path, "r") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()

                # skip blanks and pure comments
                if not line or line.startswith("#"):
                    continue

                # strip trailing inline comments
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                    if not line:
                        continue

                # require at least one '=' and split only once
                if "=" not in line:
                    # warn, but don’t crash
                    print(f"[params] Skipping line {lineno}: no '=' -> {raw!r}")
                    continue

                key, value = [s.strip() for s in line.split("=", 1)]

                # key, value = line.split("=", 1)
                # key = key.strip()
                # value = value.strip()

                self.raw_parameters[key] = value

        # -------------------
        # Output parameters
        # -------------------
        
        # Saves the simulation results in vtu files
        if "save_output" in self.raw_parameters:
            self.save_output = eval(self.raw_parameters["save_output"])

        # Saves the vtus in output_path folder
        if "output_path" in self.raw_parameters:
            self.output_path = self.raw_parameters["output_path"]

        # -------------------
        # Time parameters
        # -------------------

        if "steady" in self.raw_parameters:
            self.steady = eval(self.raw_parameters["steady"])


        if "start_time" in self.raw_parameters:
            self.start_time = float(self.raw_parameters["start_time"])

        if "final_time" in self.raw_parameters:
            self.final_time = float(self.raw_parameters["final_time"])
        
        if "time_step" in self.raw_parameters:
            self.time_step = float(self.raw_parameters["time_step"])

        # -------------------
        # Material properties
        # -------------------

        if "thermal_conductivity" in self.raw_parameters:
            self.thermal_conductivity = float(self.raw_parameters["thermal_conductivity"])
        
        if "density" in self.raw_parameters:
            self.density = float(self.raw_parameters["density"])
        
        if "heat_capacity" in self.raw_parameters:
            self.heat_capacity = float(self.raw_parameters["heat_capacity"])

        # -------------------
        # Plate dimensions
        # -------------------

        if "plate_dimensions" in self.raw_parameters:
            self.plate_dimensions = [float(i) for i in self.raw_parameters["plate_dimensions"].strip("][").split(", ")]

        # -------------------
        # Number of elements
        # -------------------

        if "nx" in self.raw_parameters:
            self.nx = int(self.raw_parameters["nx"])

        if "ny" in self.raw_parameters:
            self.ny = int(self.raw_parameters["ny"])
        
        if "nz" in self.raw_parameters:
            self.nz = int(self.raw_parameters["nz"])

        # -------------------
        # Initial conditions
        # -------------------

        # initial_condition = 0           (function of the initial temperature in the block)
        
        self.initial_condition = float(self.raw_parameters["initial_condition"])

        # -------------------------
        # Boundary conditions type
        # -------------------------

        # The following values need to be specified:
        # "dirichlet", boundary temperature
        # "neumann",   flux at the wall
        # "robin",     heat transfer coefficient and temperature at infinity
        # "no_flux"    0 
        # "surrogate", path/to/surrogate
        # "robin_with_h_from_file", path/to/adjoint_h_coefficients and temperature at infinity

        # Example in the parameter file:
        # BC_types = no_flux, no_flux, no_flux, no_flux, dirichlet, dirichlet
        # BC_values = 0, 0, 0, 0, 0, 1

        boundary_options = {"dirichlet":0, "neumann":1, "robin":2, "no_flux":3, "surrogate":4, "robin_with_h_from_file":5, "robin_natural_boundaries":6}

        if "BC_types" in self.raw_parameters:
            boundary_type_list = self.raw_parameters["BC_types"].strip().replace(" ", "").split(",")

        else:
            print("Warning: Boundary conditions were not specified.")


        if "BC_T" in self.raw_parameters:
            self.BC_T = [i for i in self.raw_parameters["BC_T"].strip().replace(" ", "").split(",")]

        if "BC_flux" in self.raw_parameters:
            self.BC_flux = [i for i in self.raw_parameters["BC_flux"].strip().replace(" ", "").split(",")]

        if "initial_convective_coefficient" in self.raw_parameters:
            self.initial_convective_coefficient = [i for i in self.raw_parameters["initial_convective_coefficient"].strip().replace(" ", "").split(",")]

        if "T_inf" in self.raw_parameters:
            self.T_inf = [i for i in self.raw_parameters["T_inf"].strip().replace(" ", "").split(",")]

        if "T_inf_natural" in self.raw_parameters:
            self.T_inf_natural = [i for i in self.raw_parameters["T_inf_natural"].strip().replace(" ", "").split(",")]

        if "natural_convective_coefficient" in self.raw_parameters:
            self.natural_convective_coefficient = [i for i in self.raw_parameters["natural_convective_coefficient"].strip().replace(" ", "").split(",")]

        self.dirichlet_boundaries = []
        self.neumann_boundaries = []
        self.robin_boundaries = []
        self.robin_natural_boundaries = []
        self.surrogate_boundaries = []
        self.robin_with_h_from_file_boundaries = []
                
        for i in range(6):

            # If the boundary type is Dirichlet, Robin or Surrogate, add the index to the list of Dirichlet BCs
            if boundary_options[boundary_type_list[i]] == 0:
                self.dirichlet_boundaries.append(i)

            # If the boundary type is Neumann, add the index to the list of Neumann BCs
            elif boundary_options[boundary_type_list[i]] == 1: 
                self.neumann_boundaries.append(i)
            
            # If the boundary type is No Flux, add the index to the list of Neumann BCs and set the flux to 0
            elif boundary_options[boundary_type_list[i]] == 3:
                self.neumann_boundaries.append(i)
                self.BC_flux[i] = "0"                

            # If The boundary type is Robin, add the index to the list of Robin BCs
            elif boundary_options[boundary_type_list[i]] == 2:
                self.robin_boundaries.append(i)
            
            # If The boundary type is Surrogate, add the index to the list of Surrogate BCs
            elif boundary_options[boundary_type_list[i]] == 4:
                self.surrogate_boundaries.append(i)

            # If the boundary type is Robin with CC from file, add the index to the list of Robin with CC from file BCs
            elif boundary_options[boundary_type_list[i]] == 5:
                self.robin_with_h_from_file_boundaries.append(i)
                if "heat_load_convective_coefficients_path" in self.raw_parameters:
                    self.heat_load_convective_coefficients_path = self.raw_parameters["heat_load_convective_coefficients_path"]
                else:
                    raise ValueError("Error: Path to the convective coefficients CSV file must be specified for 'robin_with_h_from_file' boundary condition.")

            elif boundary_options[boundary_type_list[i]] == 6:
                self.robin_natural_boundaries.append(i)

            # If boundary type does not exist, print an error message and exit
            else:
                print("Error: Boundary condition not recognized.")
                exit(1)

        if "source_term" in self.raw_parameters:
            self.source = eval(self.raw_parameters["source"])

            if self.source:
                self.source_term = self.raw_parameters["source_term"].strip().replace(" ", "")

        else:
            self.source = False

        # -------------------------
        # Solver parameters
        # -------------------------

        # TODO

        # -------------------------
        # PID parameters
        # -------------------------

        # For the PID controller, the following parameters need to be specified:
        # Apply control: apply_pid_control = True (PID controller is active), apply_pid_control = False (PID controller is inactive)
        # Control Face ID: specify the face to which the control is applied (value between 0 and 5)
        #       0: left, 1: right, 2: front, 3: back, 4: bottom, 5: top
        # Target Face ID: specify the face that should reach the target (value between 0 and 5)
        # KP: Proportional gain of the PID Controller
        # KI: Integrative gain of the PID Controller
        # KD: Derivative gain of the PID Controller
        # target: specify the target value for the target_face_id_surface

        if "apply_pid_control" in self.raw_parameters:
            self.apply_pid_control = eval(self.raw_parameters["apply_pid_control"])

        if "actuation_face_id" in self.raw_parameters:
            self.actuation_face_id = int(self.raw_parameters["actuation_face_id"])
        
        if "target_face_id" in self.raw_parameters:
            self.target_face_id = int(self.raw_parameters["target_face_id"])

        if "KP" in self.raw_parameters:
            self.KP = float(self.raw_parameters["KP"])

        if "KI" in self.raw_parameters:
            self.KI = float(self.raw_parameters["KI"])
        
        if "KD" in self.raw_parameters:
            self.KD = float(self.raw_parameters["KD"])

        if "target_temperature" in self.raw_parameters:
            self.target_temperature = [float(i) for i in self.raw_parameters["target_temperature"].strip().replace(" ", "").split(",")]

        if "optimize_target_T" in self.raw_parameters:
            self.optimize_target_T = [float(i) for i in self.raw_parameters["optimize_target_T"].strip().replace(" ", "").split(",")]

        # -------------------------
        # MPC parameters
        # -------------------------

        # For the MPC controller, the following parameters need to be specified:
        # apply_mpc_control: apply_mpc_control = True (MPC controller is active), apply_mpc_control = False (MPC controller is inactive)

        if "apply_mpc_control" in self.raw_parameters:
            self.apply_mpc_control = eval(self.raw_parameters["apply_mpc_control"])

        if "mpc_prediction_horizon" in self.raw_parameters:
            self.mpc_prediction_horizon = int(self.raw_parameters["mpc_prediction_horizon"])

        if "mpc_control_horizon" in self.raw_parameters:
            self.mpc_control_horizon = int(self.raw_parameters["mpc_control_horizon"])

        # Whether the MPC internal model is perfectly informed (default: True)

        if "mpc_informed" in self.raw_parameters:
            self.mpc_informed = eval(self.raw_parameters["mpc_informed"])
            if "mpc_reconstruction" in self.raw_parameters and self.mpc_informed == False:
                self.mpc_reconstruction = eval(self.raw_parameters["mpc_reconstruction"])
            else:
                self.mpc_reconstruction = True
        else:
            self.mpc_informed = True

        # Verbosity of the MPC controller
        if "mpc_verbose" in self.raw_parameters:
            self.mpc_verbose = eval(self.raw_parameters["mpc_verbose"])
        else:
            self.mpc_verbose = False

        # -------------------------
        # Zones
        # -------------------------

        # For the zones, the following parameters need to be specified:
        # zone: zone = True (zones may be applied to surfaces), zone = False (no zone is applied to any surface)
        # # Defines the corner coordinates for rectangular zones within the domain.
        # Each rectangle is represented as a list of 4 values: [x_min, y_min, x_max, y_max].
        # Example: zone_corners = [[0.2, 0.2, 0.3, 0.3], [0.5, 0.2, 0.6, 0.3]]
        # This parameter is parsed from the input file, where each rectangle is separated by a semicolon.
        # The parsing logic ensures each entry has exactly 4 values, otherwise raises a ValueError.

        if "zone" in self.raw_parameters:
            self.zone = eval(self.raw_parameters["zone"])

            if self.zone:
                self.zone_faces = self.raw_parameters["zone_faces"]
                if "," in self.zone_faces:
                    self.zone_faces = [int(i) for i in self.zone_faces.strip().replace(" ", "").split(",")]
                else:
                    self.zone_faces = [int(self.zone_faces)]
            
                self.zone_corners = [eval(i) if len(eval(i)) == 4 else ValueError("Zone corners are not correctly defined.") for i in self.raw_parameters["zone_corners"].strip().replace(" ", "").split(";")]

        # -------------------------
        # Flow rate echelon
        # -------------------------

        if "apply_echelon" in self.raw_parameters:
            self.apply_echelon = eval(self.raw_parameters["apply_echelon"])
            if self.apply_echelon:
                if "echelon_start_time" in self.raw_parameters:
                    self.echelon_start_time = float(self.raw_parameters["echelon_start_time"])
                if "echelon_jet_id" in self.raw_parameters:
                    self.echelon_jet_id = int(self.raw_parameters["echelon_jet_id"])
                if "echelon_value" in self.raw_parameters:
                    self.echelon_value = float(self.raw_parameters["echelon_value"])

        # -------------------------
        # Perturbation
        # -------------------------

        if "perturbation_type" in self.raw_parameters:
            self.perturbation_type = self.raw_parameters["perturbation_type"].strip()

            if getattr(self, "perturbation_type", "none").strip().lower() != "none":
                if "perturbation_time" in self.raw_parameters:
                    self.perturbation_time = float(self.raw_parameters["perturbation_time"])
                if "perturbation_face_id" in self.raw_parameters:
                    self.perturbation_face_id = int(self.raw_parameters["perturbation_face_id"])

                # Gaussian Neumann perturbation (heat flux)
                if self.perturbation_type == "gaussian_neumann":
                    if "gaussian_neumann_amplitude" in self.raw_parameters:
                        self.gaussian_neumann_amplitude = float(self.raw_parameters["gaussian_neumann_amplitude"])
                    if "gaussian_neumann_center" in self.raw_parameters:
                        self.gaussian_neumann_center = [float(i) for i in self.raw_parameters["gaussian_neumann_center"].strip("[]").split(",")]
                    if "gaussian_neumann_sigma" in self.raw_parameters:
                        self.gaussian_neumann_sigma = float(self.raw_parameters["gaussian_neumann_sigma"])

                # Gaussian Robin perturbation (convective heat transfer coefficient map)
                if self.perturbation_type == "gaussian_robin":
                    if "gaussian_robin_h_amplitude" in self.raw_parameters:
                        self.gaussian_robin_h_amplitude = float(self.raw_parameters["gaussian_robin_h_amplitude"])
                    if "gaussian_robin_center" in self.raw_parameters:
                        self.gaussian_robin_center = [float(i) for i in self.raw_parameters["gaussian_robin_center"].strip("[]").split(",")]
                    if "gaussian_robin_sigma" in self.raw_parameters:
                        self.gaussian_robin_sigma = float(self.raw_parameters["gaussian_robin_sigma"])
                    if "gaussian_robin_T_inf" in self.raw_parameters:
                        self.gaussian_robin_T_inf = float(self.raw_parameters["gaussian_robin_T_inf"])

        # -------------------------
        # Surrogate
        # -------------------------
        # Surrogate: idw_surrogate = True, To use the inverse distance weighting surrogate model
        #            nn_surrogate  = True, To use the neural network surrogate model
        # initial_inlet_configuration: initial_inlet_configuration = [0, -1, 1, -1, 0], Initial configuration of the injectors. The data is only present for 5 injectors. The value -1 is used for the outlets.
        # surrogate_path: surrogate_path = path/to/surrogate, Path to the surrogate model
        

        if "idw_surrogate"in self.raw_parameters:
            self.idw_surrogate = eval(self.raw_parameters["idw_surrogate"])
            self.initial_inlet_configuration = np.zeros(5, dtype=float)  #version for the 5x1: np.zeros(5, dtype=float)
            if 'scale_factor' in self.raw_parameters:
                self.scale_factor = float(self.raw_parameters["scale_factor"])
            else:
                self.scale_factor = 1.0

            for i, inlet_str in enumerate(self.raw_parameters["initial_inlet_configuration"].strip().replace(" ", "").split(",")):
                inlet = eval(inlet_str)
                self.initial_inlet_configuration[i] = inlet
            self.surrogate_path = self.raw_parameters["surrogate_path"]

            # Error if the number of injectors is not equal to 5
            if len(self.initial_inlet_configuration) != 5:
                raise ValueError("The number of specified injectors should be equal to 5")
            
            self.jet_diameter = float(self.raw_parameters["jet_diameter"])
            self.fluid_conductivity = float(self.raw_parameters["fluid_conductivity"])

            if "sweep_granularity" in self.raw_parameters:
                self.sweep_granularity = float(self.raw_parameters["sweep_granularity"])

        
        if "nn_surrogate" in self.raw_parameters:
            self.nn_surrogate = eval(self.raw_parameters["nn_surrogate"])
            self.initial_inlet_configuration = np.zeros(5, dtype=float)  #version for the 5x1: np.zeros(5, dtype=float)
        
        if "nn_5x1_surrogate" in self.raw_parameters:
            self.nn_5x1_surrogate = eval(self.raw_parameters["nn_5x1_surrogate"])
            self.initial_inlet_configuration = np.zeros(5, dtype=float)  #version for the 5x1: np.zeros(5, dtype=float)

        if "nn_3x3_surrogate" in self.raw_parameters:
            self.nn_3x3_surrogate = eval(self.raw_parameters["nn_3x3_surrogate"])
            self.initial_inlet_configuration = np.zeros(9, dtype=float)  #version for the 3x3: np.zeros(9, dtype=float)

        if "nn_surrogate" in self.raw_parameters or "nn_5x1_surrogate" in self.raw_parameters or "nn_3x3_surrogate" in self.raw_parameters:

            if 'scale_factor' in self.raw_parameters:
                self.scale_factor = float(self.raw_parameters["scale_factor"])
            else:
                self.scale_factor = 1.0

            for i, inlet_str in enumerate(self.raw_parameters["initial_inlet_configuration"].strip().replace(" ", "").split(",")):
                inlet = eval(inlet_str)
                self.initial_inlet_configuration[i] = inlet
            self.surrogate_path = self.raw_parameters["surrogate_path"]

            # Error if the number of injectors is not equal to 5 for the 5x1 case or 9 for the 3x3 case
            if self.nn_5x1_surrogate and len(self.initial_inlet_configuration) != 5:
                raise ValueError("The number of specified injectors should be equal to 5 for the 5x1 surrogate model")
            if self.nn_3x3_surrogate and len(self.initial_inlet_configuration) != 9:
                raise ValueError("The number of specified injectors should be equal to 9 for the 3x3 surrogate model")
            
            self.jet_diameter = float(self.raw_parameters["jet_diameter"])
            self.fluid_conductivity = float(self.raw_parameters["fluid_conductivity"])

            if "sweep_granularity" in self.raw_parameters:
                self.sweep_granularity = float(self.raw_parameters["sweep_granularity"])


        if "solve_adjoint" in self.raw_parameters:
            self.solve_adjoint = eval(self.raw_parameters["solve_adjoint"])

            if self.solve_adjoint:

                self.adjoint_max_iterations = int(self.raw_parameters["adjoint_max_iterations"])
                self.adjoint_tolerance = float(self.raw_parameters["adjoint_tolerance"])
                self.adjoint_target_face = int(self.raw_parameters["adjoint_target_face"])

                self.adjoint_optimize_face = int(self.raw_parameters["adjoint_optimize_face"])
                self.adjoint_step_size = float(self.raw_parameters["adjoint_step_size"])
                self.adjoint_track_error = eval(self.raw_parameters["adjoint_track_error"])
                self.adjoint_output_results = eval(self.raw_parameters["adjoint_output_results"])

                self.adjoint_target_array_from_file = eval(self.raw_parameters["adjoint_target_array_from_file"])
                
                if self.adjoint_target_array_from_file:
                    self.adjoint_target_array_file = self.raw_parameters["adjoint_target_array_file"]
                    
                    # The columns in the csv represent the target temperature on the surface of the plate. In the CSV file, every column is a temperature on a whole plate flattened (order="F"). For now, the simulation number of elements on the target face is hardcoded to be equal to the number of elements in the plate.
                    
                    # Read the csv file and store the data in a numpy array of dimensions (n_columns in csv, n_rows in csv). The number of rows in the csv file is equal to the number of elements in the target face. The number of columns in the csv file is equal to the number of time steps in the problem. This is hardcoded for now.

                    # Read csv into numpy array
                    self.adjoint_target_array = np.genfromtxt(self.adjoint_target_array_file, delimiter=',')

                elif "adjoint_target_array" in self.raw_parameters:
                    # Get dimension number of dofs in the target face
                    N = self.get_n_dofs_face(self.adjoint_target_face)
                    self.adjoint_target_array = np.zeros(N)
                    target_str = self.raw_parameters["adjoint_target_array"]
                    self.adjoint_target_array = eval(target_str)
                
                elif "adjoint_target_T" in self.raw_parameters:
                    self.adjoint_target_T = float(self.raw_parameters["adjoint_target_T"])

    def get_n_dofs_face(self, face_id):
        if face_id in [0, 1]:
            return self.nz * self.ny
        elif face_id in [2, 3]:
            return self.nz * self.nx
        elif face_id in [4, 5]:
            return self.nx * self.ny
        else:
            raise ValueError("Invalid face id")


            