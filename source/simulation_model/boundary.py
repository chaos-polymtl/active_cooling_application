# from anyio import current_time
import numpy as np
import glob
import os
from scipy.interpolate import RegularGridInterpolator
from src.utility import *
import torch 
import torch.nn as nn
from torch.nn.functional import tanh, leaky_relu


class Boundary:
    """
    Class to calculate the boundary conditions.
    """

    def __init__(self, params, points, dx, dy, dz):
        """
        Initialize the Boundary class.
        AttributeError: 'Boundary' object has no attribute 'flatten_index'
        :param params: Parameters object
        """
        self.params = params
        self.dx, self.dy, self.dz = dx, dy, dz
        self.points = points

        self.dict_boundary_points = compute_boundary_dict(params.nx, params.ny, params.nz)
        self.convective_coefficient = self.set_initial_convective_coefficient()
        
        if params.zone:
            self.compute_zones()

        self.BC_T = {}
        self.BC_flux = {}
        self.T_inf = {}

        self.inlet_configuration = params.initial_inlet_configuration

        # Load the idw surrogate data if used
        if params.idw_surrogate:
            self.injector_state, self.data = self.load_idw_data(params.surrogate_path)
        if params.nn_surrogate:
            self.nn_surrogate, self.scaler, self.nn_pixel_per_row, self.nn_pixel_per_cols = self.load_nn_surrogate(params.surrogate_path)
        if params.nn_5x1_surrogate:
            self.nn_surrogate = self.load_5x1_surrogate(params.surrogate_path)
            self.scaler = None
        if params.nn_3x3_surrogate:
            self.nn_surrogate = self. load_3x3_surrogate(params.surrogate_path)
            self.scaler = None
        
        self.iteration = 0

        self.evaluate_boundary_conditions(0)

    def set_inlet_configuration(self, new_configuration):
        self.inlet_configuration = new_configuration

    def set_rhs(self, face_id, rhs, new_rhs):
        """
        Set the right hand side for a given face.
        """
        for i, index in enumerate(self.dict_boundary_points[face_id]):
            rhs[index] = new_rhs[i]

    def set_A(self, face_id, A, new_A):
        """
        Set the right hand side for a given face.
        """
        previous_A = A.data
        for i, index in enumerate(self.dict_boundary_points[face_id]):
            A.rows[index] = [index]
            A.data[index] = [previous_A[index]]

    def set_convective_heat_transfer_map_for_face(self, face_id, convective_coefficient_map):
        """
        Set the convective heat transfer coefficient map for a given face.
        """
        for i, index in enumerate(self.dict_boundary_points[face_id]):
            self.convective_coefficient[index] = convective_coefficient_map[i]

    def get_convective_coefficient_at_face(self, face_id):
        """
        Get the convective coefficient at a given face.
        """
        face_id = int(face_id)
        indices = self.dict_boundary_points[face_id]
        return np.array([self.convective_coefficient.get(int(idx), 0.0) for idx in indices], dtype=float)


    def load_idw_data(self, csv_folder):
        """
        Load the IDW data from the csv files. Every dataframe is the result of a single simulation. The weighted sum of these simulations will be used to interpolate the heat flux on the boundary for a given state of the injectors. 
        """

        # Get all the csv files from the folder
        csv_files = sorted(glob.glob(os.path.join(csv_folder, '*.csv')))

        # Crash if there are no csv files
        if len(csv_files) == 0:
            raise ValueError("No csv files for the inverse distance weighted algorithm found in the folder")
        
        # Load the data from the csv files
        data, injector_state = [], []
        for csv_file in csv_files:

            # Append the data to the csv list
            data.append(np.genfromtxt(csv_file, delimiter='\t', skip_header=3)[:, 1:])
            
            # TODO Need to find a better way to get the injector state from the csv files (maybe write it directly in them according to the specified injector array shape in the prm file instead of putting into the name of the file)
            injector_state.append(np.array([float(csv_file[:-4].split("_")[-5:][i]) for i in range(5)]))



        # Check if the data loaded from the csv files have the same shape for the interpolation
        shapes = [data[i].shape for i in range(len(data))]
        if len(set(shapes)) != 1:
            raise ValueError("The data loaded from the csv files have different shapes. The dimensions are not consistent.") 

        # Add all the mirrored datasets
        flipped_injector_state = [np.flip(injector_state[i], axis=0) for i in range(len(injector_state))]
        flipped_data           = [np.flip(data[i], axis=1)           for i in range(len(data))]
        
        injector_state += flipped_injector_state
        data += flipped_data
                
        # Transform the data into numpy arrays
        injector_state = np.array(injector_state)
        data = np.array(data)

        # we replace injector_state outlets by 100 000 so different configurations are heavily penalized
        injector_state[injector_state == -1] = 100000

        return injector_state, data
    
    def load_nn_surrogate(self, nn_surrogate_path):
        """
        Load the neural network surrogate. from a .pkl file containing
        1. The weights of the neural network
        2. The scaller used to normalize the data
        3. The number of pixels per row. The nn predicts the heat flux on a vector. This information is useful to reshape it back 2 a 2D array. 
        4. The number of pixels per column
        """
        # Load the surrogate model from the .pkl file        
        device = torch.device("cpu")
        model = torch.load(nn_surrogate_path, map_location=device, weights_only=False)

        model_state_dict = model["model_state"]
        scaler = model["scaler"]
        nn_pixel_per_row = model["n_rows"]
        nn_pixel_per_col = model["n_cols"]

        # Extract information from the loaded model create the neural network
        input_layer_size = int(model_state_dict['fc_in.weight'].size()[1])
        output_layer_size = int(model_state_dict['fc_out.weight'].size()[0])        
        hidden_layer_size = int(model_state_dict['fc_in.weight'].size()[0])
        hidden_layer_width = int((len(model_state_dict) - 4)/2) # 4 is for the input and output layers. We divide by two for the bias and the weights

        # Create the neural network
        nn_surrogate = heat_predictor(input_layer_size, output_layer_size, hidden_layer_size, hidden_layer_width, device)

        nn_surrogate.load_state_dict(model_state_dict)

        nn_surrogate.eval()

        return nn_surrogate, scaler, nn_pixel_per_row, nn_pixel_per_col

    def load_5x1_surrogate(self, nn_surrogate_path):
        """
        Function to load the surrogate model. We do not return the scaler as this surrogate does not use it.
        """
        # Load the model from the path
        low_re_model = torch.load(nn_surrogate_path, weights_only=False)
        
        # Load the model
        model_state = low_re_model["model_state_dict"]
        
        net = Model_5x1()
        
        net.load_state_dict(model_state)
        net.eval()
        
        return net
    
    def load_3x3_surrogate(self, nn_surrogate_path):
        """
        Function to load the 3x3 surrogate model. We do not return the scaler as this surrogate does not use it.
        """
        # Load the model from the path
        model_data = torch.load(nn_surrogate_path, weights_only=False)
        
        # Load the model
        model_state = model_data["model_state_dict"]
        
        net = Model_3x3()
        
        net.load_state_dict(model_state)
        net.eval()
        
        return net

    def compute_idw_h_map(self, injector_state: np.ndarray, n_averaged_configurations=5):
        """
        INPUT : the current state of the injectors is a list of 0 (closed) to 1 (completely opened) values or -1 for outlets. The shape of the injector array is specified in the prm file -> 5, 1 for a line of injectors. 

        RETURNS : A 2D numpy array corresponding to the heat_flux on the top of the plate. It uses the same discretization as the one used in the csv files.

        """
        current_injector_state = injector_state.copy()
        if type(current_injector_state) != np.ndarray:
            raise TypeError("Current injector state should be a numpy array")

        # Replace all the -1 by 100000 so the bad configuration is heavily penalized
        current_injector_state[current_injector_state == -1] = 100000

        # Compute distance between the configurations and the current_injector_state
        abs_distance = np.sum(np.abs(self.injector_state - current_injector_state), axis = 1)

        # Compute the index of the IDW using the 5 (HARDCODED) closest configurations
        lowest_indices = np.argsort(abs_distance)[:n_averaged_configurations]

        # Non-zero values
        abs_distance = np.where(abs_distance[lowest_indices]==0, 1e-8, abs_distance[lowest_indices])

        # Compute the weigths using the inverse distance
        idw_weights = (1/abs_distance) / np.sum(1/abs_distance)
        
        # Compute the flux using the IDW
        training_heat_flux = np.sum(self.data[lowest_indices] * idw_weights[:, np.newaxis, np.newaxis], axis=0)

        # Rescale the data from dimensionless to dimension
        training_k = 0.0045146
        training_delta_T = 30.
        training_L = 1.

        # MAYBE ERROR - The parameters to dimensionalize are also applied after in the code
        convective_coefficient_map = training_heat_flux * self.params.fluid_conductivity * training_L / (training_k * training_delta_T * self.params.jet_diameter)

        return convective_coefficient_map

    def compute_nn_nusselt_map(self, current_injector_state):
        """
        Compute the heat flux map using the neural network surrogate.
        """
                
        input = torch.tensor(current_injector_state.copy(), dtype=torch.float32).to(torch.device('cpu'))

        output = self.nn_surrogate(input)

        if self.scaler is not None:
            # Rescale the data using the scaler
            output = self.scaler.inverse_transform(output.detach().cpu().numpy().reshape(self.nn_pixel_per_row, self.nn_pixel_per_cols))
        
        # Rescale the data
        return output 

    def evaluate_surrogate(self, n_averaged_configurations=5):
        """
        Update the convective coefficient for a certain inlet configuration using the IDW or the NN surrogate.
        """
        
        # Rescale the data from dimensionless to dimension
        training_k = 0.0096452
        training_delta_T = 30.
        training_L = 1  

        if self.params.idw_surrogate:
            h_map = self.compute_idw_h_map(self.inlet_configuration, n_averaged_configurations)
            # MAYBE ERROR - see comment in compute_idw_h_map. Also training_delta_T should not be applied. Not changed for the tests
            convective_coefficient_map = h_map * self.params.fluid_conductivity * training_L / (training_k * training_delta_T * self.params.jet_diameter) * self.params.scale_factor

        elif self.params.nn_surrogate:
            nusselt_map = self.compute_nn_nusselt_map(self.inlet_configuration)
            convective_coefficient_map = nusselt_map * (self.params.fluid_conductivity / (self.params.jet_diameter)) * self.params.scale_factor

        elif self.params.nn_5x1_surrogate:
            
            # The input is transformed for the 5x1 surrogate model
            X = []
            x_pos_mask =np.array([[-1, -0.5, 0, 0.5, 1]]) 
            inlet_outlet_mask = np.array([[-1.0 if float(i) == -1  else 1.0 for i in self.inlet_configuration]])
            configuration = np.array([[0.0 if float(i) == -1 else float(i) for i in self.inlet_configuration]])  # Replace -1 with 0.0

            # We append the configuration and the other masks to the X list
            X.append(np.stack([configuration, inlet_outlet_mask, x_pos_mask]))
            
            # Predict and transform the Nusselt into a convective coefficient map
            nusselt_map = self.nn_surrogate(torch.tensor(X, dtype=torch.float32)).detach().cpu().numpy()[0][0]
            convective_coefficient_map = nusselt_map * (self.params.fluid_conductivity / (self.params.jet_diameter)) * self.params.scale_factor

        elif self.params.nn_3x3_surrogate:

            # The input is transformed for the 3x3 surrogate model

            # Reshape the 9 jet values to a 3x3 array
            conf = np.asarray(self.inlet_configuration, dtype=float).ravel()
            if conf.size != 9:
                raise ValueError(f"nn_3x3_surrogate expects 9 injector values, got {conf.size}")
            conf_grid = conf.reshape((3, 3))

            X = []

            x_row = np.array([[-1.0, 0.0, 1.0]]) # (1, 3)
            z_col = np.array([[-1.0], [0.0], [1.0]]) # (3, 1)
            x_pos_mask = np.tile(x_row, (3, 1)) # (3, 3)
            z_pos_mask = np.tile(z_col, (1, 3)) # (3, 3)

            inlet_outlet_mask = np.where(conf_grid<0.0, -1.0, 1.0) # (3, 3)
            configuration = np.where(conf_grid<0.0, 0.0, conf_grid)  # Replace -1 with 0.0 -> (3, 3)

            # We append the configuration and the other masks to the X list
            X.append(np.stack([configuration, inlet_outlet_mask, x_pos_mask, z_pos_mask], axis=0)) # (4, 3, 3)

            # We convert the list to a numpy array
            X = np.array(X, dtype=np.float32)  # make it (1, 4, 3, 3)

            # Predict and transform the Nusselt into a convective coefficient map
            nusselt_map = self.nn_surrogate(torch.tensor(X, dtype=torch.float32)).detach().cpu().numpy()[0][0]
            convective_coefficient_map = nusselt_map * (self.params.fluid_conductivity / (self.params.jet_diameter)) * self.params.scale_factor

        else:
            return
        
        # We create an interpolator to be able to extract the convective coefficient value at the boundary    
        for id in self.params.surrogate_boundaries:
            dimension_indices = [0, 1, 2]
            del dimension_indices[id//2]
            x = np.linspace(0, self.params.plate_dimensions[dimension_indices[0]], convective_coefficient_map.shape[1])
            y = np.linspace(0, self.params.plate_dimensions[dimension_indices[1]], convective_coefficient_map.shape[0])

            interpolator = RegularGridInterpolator((x, y), convective_coefficient_map.T)
        
            for index in self.dict_boundary_points[id]:
                x, y, z = self.points[index]
                self.convective_coefficient[index] = interpolator([x, y])[0]
                self.T_inf[index] = eval(self.params.T_inf[id])

    def evaluate_boundary_conditions(self, current_time):
        """
        Evaluate the boundary conditions.
        """

        t = current_time

        for id in self.params.dirichlet_boundaries:
            if type(self.params.BC_T[id]) == str:

                for index in self.dict_boundary_points[id]:
                    x, y, z = self.points[index]
                    
                    # Commente pour mettre une température sur mesure
                    self.BC_T[index] = eval(self.params.BC_T[id])
                
            elif len(self.params.BC_T[id]) >= 1:
                current_id_points = self.dict_boundary_points[id]
                for i, index in enumerate(current_id_points):
                    self.BC_T[index] = self.params.BC_T[id][i]

            else:
                exit("Dirichlet boundary condition not recognized")
        
        for id in self.params.neumann_boundaries:
            for index in self.dict_boundary_points[id]:
                x, y, z = self.points[index]

                self.BC_flux[index] = eval(self.params.BC_flux[id])

        for id in self.params.robin_boundaries + self.params.robin_natural_boundaries:

            for index in self.dict_boundary_points[id]:
                x, y, z = self.points[index]

                self.T_inf[index] = eval(self.params.T_inf[id])

                # Commented this section. Caused the convective coefficient was reinitialized in the adjoint problem. We already set the initial convective coefficient in the  
                # if t == 0 or "t" in self.params.initial_convective_coefficient[id]:
                    # self.convective_coefficient[index] = eval(self.params.initial_convective_coefficient[id])

    def assemble_boundaries(self, A, rhs, current_time, sensitivity=False, direct_temp=None, d_k=None):

        # # Ensure time-dependent BC expressions are refreshed each step
        # self.evaluate_boundary_conditions(current_time)

        # Boundary conditions
        self.evaluate_surrogate()

        self.apply_neumann_boundary_conditions(A, rhs)
        if not sensitivity:
            self.apply_robin_boundary_conditions(A, rhs)
        else:
            if direct_temp is not None and d_k is not None:
                self.apply_sensitivity_robin_boundary_conditions(A, rhs, direct_temp, d_k)
            else:
                raise ValueError("Direct temperature and d_k must be provided for sensitivity analysis.")
        self.apply_dirichlet_boundary_conditions(A, rhs)

    def assemble_boundaries_ss(self, A, rhs):
        
        # Boundary conditions
        self.evaluate_surrogate()

        self.apply_neumann_boundary_conditions(A, rhs)
        self.apply_robin_boundary_conditions(A, rhs)
        self.apply_dirichlet_boundary_conditions(A, rhs)

    def assemble_boundaries_adjoint(self, A, rhs, current_time):

        # Boundary conditions
        self.evaluate_boundary_conditions(current_time)
        self.evaluate_surrogate()

        self.apply_neumann_boundary_conditions(A, rhs)
        self.apply_adjoint_robin_boundary_conditions(A, rhs)
    
    def compute_normal(self, boundary_id):
        """
        Return the normal index for a given face
        """

        if boundary_id == 0:
            return -1, 0, 0
        elif boundary_id == 1:
            return 1, 0, 0
        elif boundary_id == 2:
            return 0, -1, 0
        elif boundary_id == 3:
            return 0, 1, 0
        elif boundary_id == 4:
            return 0, 0, -1
        elif boundary_id == 5:
            return 0, 0, 1
        
    def compute_zones(self):
        """
        Apply zones to the model.
        """
        zone_corners = self.params.zone_corners
        self.points_in_zone = {}
        boundary_points = self.dict_boundary_points
        n_zones = len(zone_corners)

        for face in self.params.zone_faces:
            self.points_in_zone[face] = {}
            dimension_indices = [0, 1, 2]
            del dimension_indices[face//2]

            for i in range(n_zones):
                self.points_in_zone[face][i] = []
                for index in boundary_points[face]:
                    point = self.points[index]
                    if (point[dimension_indices[0]] >= zone_corners[i][0] and 
                        point[dimension_indices[0]] <= zone_corners[i][2] and 
                        point[dimension_indices[1]] >= zone_corners[i][1] and 
                        point[dimension_indices[1]] <= zone_corners[i][3]):
                        self.points_in_zone[face][i].append(index)
    
    def apply_dirichlet_boundary_conditions(self, A, rhs):
        """
        Apply Dirichlet boundary conditions to Dirichlet faces.
        """

        for id in self.params.dirichlet_boundaries:
            for index in self.dict_boundary_points[id]:
                A[index, :] = 0
                A[index, index] = 1

                rhs[index] = self.BC_T[index]

    def apply_neumann_boundary_conditions(self, A, rhs):
        """
        Apply Neumann boundary conditions to Neumann faces.
        """ 

        nx, ny = self.params.nx, self.params.ny

        for id in self.params.neumann_boundaries:
            for index in self.dict_boundary_points[id]:

                i, j, k = self.compute_normal(id)
                space_between_neighbours = np.abs(i*self.dx + j*self.dy + k*self.dz)

                beta = space_between_neighbours / self.params.thermal_conductivity

                A.rows[index] = [index, compute_neighbor(index, id, nx, ny)]
                A.data[index] = [1.0, -1.0]

                rhs[index] = beta * self.BC_flux[index]

    def apply_robin_boundary_conditions(self, A, rhs):
        """
        Apply Robin boundary conditions to Robin faces.
        """

        nx, ny = self.params.nx, self.params.ny
        # Set a convective coefficient based on the boundary id and the file
        for id in self.params.robin_with_h_from_file_boundaries:
            # This is important, if we are in transient, we MUST have one convective coefficient per time step, including the initial condition. For now, the files MUST be named prefix_0.csv, prefix_1.csv, etc. Use heat_load_convective_coefficient_path = ./tests/bc_heat_load_from_file/adjoint_h_conv (without the _i.csv at the end. The code appends it)
            self.convective_coefficients_from_file = self.compute_convective_coefficients_from_file(self.params.heat_load_convective_coefficients_path)
            # else:
            #     self.convective_coefficients_from_file = self.compute_convective_coefficients_from_file(self.params.heat_load_convective_coefficients_path + "_" + str(self.iteration) + ".csv")

            self.iteration += 1
            for index in self.dict_boundary_points[id]:
                x, y, z = self.points[index]

                self.T_inf[index] = eval(self.params.T_inf[id])
                self.convective_coefficient[index] = self.convective_coefficients_from_file[index]

        for id in self.params.robin_boundaries + self.params.surrogate_boundaries + self.params.robin_natural_boundaries + self.params.robin_with_h_from_file_boundaries:
            has_natural = id in self.params.robin_natural_boundaries
            if has_natural:
                h_natural = float(self.params.natural_convective_coefficient[id])
                T_inf_natural = float(self.params.T_inf_natural[id])
            for index in self.dict_boundary_points[id]:

                i, j, k = self.compute_normal(id)
                space_between_neighbours = np.abs(i*self.dx + j*self.dy + k*self.dz)

                beta = -self.convective_coefficient[index] * space_between_neighbours / self.params.thermal_conductivity

                A.rows[index] = [index, compute_neighbor(index, id, nx, ny)]
                A.data[index] = [1 - beta, -1.0]

                rhs[index] = - beta * self.T_inf[index]

                if has_natural:
                    beta_natural = -h_natural * space_between_neighbours / self.params.thermal_conductivity
                    A[index, index] += -beta_natural
                    rhs[index] += - beta_natural * T_inf_natural

                # Add a flux (ex. gaussian heat load) to the rhs
                q_value = self.BC_flux.get(index, 0.0)
                if q_value != 0.0:
                    beta_flux = space_between_neighbours / self.params.thermal_conductivity
                    rhs[index] += beta_flux * q_value

    def _face_uv_axes(self, face_id: int):
        """
        Return the two in-plane coordinate indices (u,v) for this face.
        Uses same convention as elsewhere: face_id//2 gives the normal axis.
        """
        axes = [0, 1, 2]
        del axes[face_id // 2]          # remove normal axis -> keep the 2 tangential axes
        return axes[0], axes[1]         # (u_idx, v_idx)
    

    def add_gaussian_neumann(self, face_id: int, center_xy, sigma: float, amplitude: float, superpose: bool = True):
        """
        Neumann Gaussian on a face:
        q''(u,v) = amplitude * exp( -((u - u0)^2 + (v - v0)^2) / (2*sigma^2) )
        Writes into self.BC_flux (W/m^2) for indices on the selected face.
        """
        u0, v0 = float(center_xy[0]), float(center_xy[1])
        u_idx, v_idx = self._face_uv_axes(face_id)
        sig2 = sigma**2
        eps = 1e-300  # guard for underflow in extreme tails

        for idx in self.dict_boundary_points[face_id]:
            u = self.points[idx][u_idx]
            v = self.points[idx][v_idx]
            r2 = (u - u0)**2 + (v - v0)**2
            G = float(amplitude * np.exp(-0.5 * r2 / sig2))
            if superpose and (idx in self.BC_flux):
                self.BC_flux[idx] += G
            else:
                self.BC_flux[idx] = G

    def add_gaussian_robin(self, face_id: int, center_xy, sigma: float, h_peak: float, T_inf_jet: float, superpose: bool = True):
        """
        Robin Gaussian on a face:
            h(u,v) = h_peak * exp( -((u - u0)^2 + (v - v0)^2) / (2*sigma^2) )
            q = h * (T_surface - T_inf_jet)
        Writes into self.convective_coefficient (W/m^2-K) and self.T_inf (temperature units).
        """
        u0, v0 = float(center_xy[0]), float(center_xy[1])
        u_idx, v_idx = self._face_uv_axes(face_id)
        sig2 = sigma ** 2
        thresh = 1e-12 * max(1.0, abs(h_peak))  # where h is “non-negligible”

        for idx in self.dict_boundary_points[face_id]:
            u = self.points[idx][u_idx]
            v = self.points[idx][v_idx]
            r2 = (u - u0)**2 + (v - v0)**2
            h_val = float(h_peak * np.exp(-0.5 * r2 / sig2))

            if not superpose:
                # Replace: set h exactly to the Gaussian map; set T_inf uniformly where active
                self.convective_coefficient[idx] = h_val
                if h_val > thresh:
                    self.T_inf[idx] = float(T_inf_jet)
                else:
                    # leave previous T_inf if h ~ 0 (node effectively not Robin)
                    self.T_inf[idx] = self.T_inf.get(idx, float(T_inf_jet))
            else:
                # Superpose h fields; keep T_inf = jet where h contributes
                self.convective_coefficient[idx] = self.convective_coefficient.get(idx, 0.0) + h_val
                if h_val > thresh:
                    self.T_inf[idx] = float(T_inf_jet)

    def apply_gaussian_perturbation_from_params(self):
        """
        Reads the ParametersHandler fields listed and applies the one configured perturbation.
        Call this once when it should *start* (e.g., when time >= params.perturbation_time).
        """
        if self.params.perturbation_type is None:
            return

        ptype = str(self.params.perturbation_type).lower()
        face  = int(self.params.perturbation_face_id)

        if ptype == "gaussian_neumann":
            self.add_gaussian_neumann(
                face_id   = face,
                center_xy = self.params.gaussian_neumann_center,
                sigma     = self.params.gaussian_neumann_sigma,
                amplitude = self.params.gaussian_neumann_amplitude,
                superpose = True,
            )
        elif ptype == "gaussian_robin":
            self.add_gaussian_robin(
                face_id   = face,
                center_xy = self.params.gaussian_robin_center,
                sigma     = self.params.gaussian_robin_sigma,
                h_peak    = self.params.gaussian_robin_h_amplitude,
                T_inf_jet = self.params.gaussian_robin_T_inf,
                superpose = True,
            )

    def apply_sensitivity_robin_boundary_conditions(self, A, rhs, direct_temp, d_k):
        """
        Apply Robin boundary conditions to Robin faces for the SENSITIVITY problem.
        """

        nx, ny = self.params.nx, self.params.ny

        for id in self.params.robin_boundaries + self.params.surrogate_boundaries + self.params.robin_natural_boundaries + self.params.robin_with_h_from_file_boundaries:
            if self.params.adjoint_target_face == id:
                for index_2d, index in enumerate(self.dict_boundary_points[id]):

                    i, j, k = self.compute_normal(id)
                    space_between_neighbours = np.abs(i*self.dx + j*self.dy + k*self.dz)

                    A.rows[index] = [index, compute_neighbor(index, id, nx, ny)]
                    A.data[index] = [1 - space_between_neighbours / self.params.thermal_conductivity * (self.convective_coefficient[index] + d_k[index_2d]) , -1.0]

                    rhs[index] = space_between_neighbours / self.params.thermal_conductivity * (direct_temp[index_2d] - eval(self.params.T_inf[id])) * d_k[index_2d]

            else:
                for index in self.dict_boundary_points[id]:

                    i, j, k = self.compute_normal(id)
                    space_between_neighbours = np.abs(i*self.dx + j*self.dy + k*self.dz)

                    beta = -self.convective_coefficient[index] * space_between_neighbours / self.params.thermal_conductivity

                    A.rows[index] = [index, compute_neighbor(index, id, nx, ny)]
                    A.data[index] = [1 - beta, -1.0]

                    rhs[index] = - beta * self.T_inf[index]

    def  apply_adjoint_robin_boundary_conditions(self, A, rhs):
        """
        Apply Robin boundary conditions to Robin faces for the UNSTEADY adjoint problem.
        """

        nx, ny = self.params.nx, self.params.ny

        for id in self.params.robin_boundaries + self.params.surrogate_boundaries + self.params.robin_natural_boundaries + self.params.robin_with_h_from_file_boundaries:
        
            for index in self.dict_boundary_points[id]:

                i, j, k = self.compute_normal(id)
                space_between_neighbours = np.abs(i*self.dx + j*self.dy + k*self.dz)

                beta = -self.convective_coefficient[index] * space_between_neighbours / self.params.thermal_conductivity

                A.rows[index] = [index, compute_neighbor(index, id, nx, ny)]
                A.data[index] = [1 + beta, -1.0]

                rhs[index] = 0 #is equal to 0 for the adjoint

    def set_initial_convective_coefficient(self):
        """
        Compute the convective coefficients for the Robin boundary conditions.
        """
        convective_coefficient = {}

        for id in self.params.robin_boundaries:
            for index in self.dict_boundary_points[id]:
                x, y, z = self.points[index]
                convective_coefficient[index] = eval(self.params.initial_convective_coefficient[id])
        
        return convective_coefficient

    def compute_convective_coefficients_from_file(self, heat_load_convective_coefficients_path):
        """
        Set the convective coefficient for the Robin boundary conditions from a file. The Robin BC comes from a .csv file. If steady=True, the convective coefficient is the last line. If the simulation is unsteady, we use the line corresponding to the current time step.
        
        Compute the convective coefficients for the Robin boundary conditions from a file.
        """

        convective_coefficient = {}

        # every line should contain the convective coefficient for each point in the domain. 
        h_coefficients = np.genfromtxt(heat_load_convective_coefficients_path, delimiter=',')

        # if array is one-dimensional, we need to reshape it
        if len(h_coefficients.shape) == 1:
            h_coefficients = np.reshape(h_coefficients, (1, -1))

        if self.params.steady:
            # Get the last line of the CSV file
            h_coefficients = h_coefficients[-1, :]
        else:
            # Get the line corresponding to the current time step
            h_coefficients = h_coefficients[self.iteration, :]

        # Get the convective coefficients for each face
        for id in self.params.robin_with_h_from_file_boundaries:
            for i, index in enumerate(self.dict_boundary_points[id]):                 
                convective_coefficient[index] = h_coefficients[i]

        return convective_coefficient
    
    def reset_boundary(self, face_id, new_h=0.0, new_T_inf=None):
        """
        Reset the convective coefficient and T_inf on a given face to new uniform values.
        Used for uninformed MPC initialization before heat load reconstruction.
        """
        face_indices = self.dict_boundary_points[face_id]

        # Determine T_inf
        if new_T_inf is None:
            try:
                new_T_inf_val = eval(self.params.T_inf[face_id])
            except Exception:
                new_T_inf_val = float(self.params.T_inf[face_id])
        else:
            new_T_inf_val = float(new_T_inf)

        for idx in face_indices:
            self.convective_coefficient[idx] = float(new_h)
            self.T_inf[idx] = new_T_inf_val

        print(f"[Boundary] Reset face {face_id}: h = {new_h}, T_inf = {new_T_inf_val}.")
    
    def apply_reconstructed_h(self, face_id, h_array, new_T_inf=None):
        """
        Apply a reconstrcuted convective coefficient array to a given face.
        h_array: 1D array of convective coefficient values for the face.
        new_T_inf: Optional uniform temperature to set for T_inf on this face.
        """
        h_array = np.asarray(h_array, dtype=float).flatten()
        face_indices = self.dict_boundary_points[face_id]
        n_face = len(face_indices)

        # Sanity check
        if h_array.size != n_face:
            raise ValueError(f"[Boundary] h_array length mismatch: got {h_array.size}, expected {n_face} for face {face_id}.")

        # Determine T_inf
        if new_T_inf is None:
            try:
                new_T_inf_val = eval(self.params.T_inf[face_id])
            except Exception:
                new_T_inf_val = float(self.params.T_inf[face_id])
        else:
            new_T_inf_val = float(new_T_inf)

        # Assign reconstructed h and T_inf
        for i, index in enumerate(face_indices):
            self.convective_coefficient[index] = float(h_array[i])
            self.T_inf[index] = new_T_inf_val
        
        print(f"[Boundary] Applied reconstructed h to face {face_id}.")

class heat_predictor(nn.Module):
    def __init__(self, x_size, y_size, SIZE, NUMBER_OF_HIDDEN_LAYER, device):
        super(heat_predictor, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc_in = torch.nn.Linear(x_size, SIZE).to(device)
        self.fcs = nn.ModuleList([torch.nn.Linear(SIZE, SIZE).to(device) for _ in range(NUMBER_OF_HIDDEN_LAYER)])
        self.fc_out = torch.nn.Linear(SIZE, y_size).to(device)

    def forward(self, x):
        x = self.fc_in(x)
        for fct in self.fcs:
          x = self.dropout(tanh(fct(x)))
        x = leaky_relu(self.fc_out(x))
        return x

class Model_3x3(nn.Module):
    def __init__(self, input_dim=4, max_channels=64, max_image_dim=256, min_image_dim=128, channel_scaling=1, pooling=True, dropout_p=0.0):
        super().__init__()

        # make sure max_image_dim >= min_image_dim
        if max_image_dim <= min_image_dim:
            raise ValueError("max_image_dim must be greater than min_image_dim")

        # Parameters
        self.input_dim = input_dim
        self.max_channels = max_channels
        self.max_image_dim = max_image_dim
        self.min_image_dim = min_image_dim
        self.channel_scaling = channel_scaling
        self.pooling = pooling
        self.init_image_dim = 3  # Assumed input spatial dimension (HxW = 3x3) - HARDCODED

        # Layers
        self.initial_deconv = self._make_initial_deconv()
        self.deconvolutions = self._make_deconvolution_layers()
        self.convolutions = self._make_convolution_layers()
        self.downsample = nn.Upsample(size=(75, 75), mode='bilinear', align_corners=False) # - HARDCODED

        # Activations and dropout
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout2d(p=dropout_p)
        
        # Testing slightly different architecture
        self.transposed_conv = nn.ConvTranspose2d(4, 64, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)

    def _make_initial_deconv(self):
        in_channels = self.input_dim
        out_channels = in_channels * (2 ** self.channel_scaling)
        self.current_channels = out_channels
        self.current_image_dim = 8  # Resulting image size after first deconv - HARDCODED
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2)

    def _make_deconvolution_layers(self):
        layers = nn.ModuleList()
        while self.current_image_dim < self.max_image_dim:
            in_ch = self.current_channels
            out_ch = min(in_ch * (2 ** self.channel_scaling), self.max_channels)
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            self.current_channels = out_ch
            self.current_image_dim *= 2
        return layers

    def _make_convolution_layers(self):
        layers = nn.ModuleList()
        while self.current_image_dim > self.min_image_dim:
            in_ch = self.current_channels
            self.current_image_dim //= 2
            out_ch = 1 if self.current_image_dim <= self.min_image_dim else max(in_ch // (2 ** self.channel_scaling), 1)
            if self.pooling:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))

            self.current_channels = out_ch
        return layers

    def forward(self, x):

        x = self.initial_deconv(x)

        for deconv in self.deconvolutions:
            x = deconv(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
        
        for conv in self.convolutions:
            x = conv(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)

        x = self.downsample(x)

        return x

class Model_5x1(nn.Module):
    def __init__(self, input_dim=3, max_channels=256, max_image_dim=256, min_image_dim=128, channel_scaling=1, pooling=True, dropout_p=0.0):
        super().__init__()

        # make sure max_image_dim >= min_image_dim
        if max_image_dim <= min_image_dim:
            raise ValueError("max_image_dim must be greater than min_image_dim")

        # Parameters
        self.input_dim = input_dim
        self.max_channels = max_channels
        self.max_image_dim = max_image_dim
        self.min_image_dim = min_image_dim
        self.channel_scaling = channel_scaling
        self.pooling = pooling
        self.init_image_dim = 5  # Assumed input spatial dimension (HxW = 3x3) - HARDCODED

        # Layers
        self.initial_deconv = self._make_initial_deconv()
        self.deconvolutions = self._make_deconvolution_layers()
        self.convolutions = self._make_convolution_layers()
        self.downsample = nn.Upsample(size=(25, 101), mode='bilinear', align_corners=False) # - HARDCODED

        # Activations and dropout
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def _make_initial_deconv(self):
        in_channels = self.input_dim
        out_channels = 8 # HARDCODED
        self.current_channels = out_channels
        self.current_image_dim = 8  # Resulting image size after first deconv - HARDCODED
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 4), stride=(1, 2), padding=(0, 2))

    def _make_deconvolution_layers(self):
        layers = nn.ModuleList()
        while self.current_image_dim < self.max_image_dim:
            in_ch = self.current_channels
            out_ch = min(in_ch * (2 ** self.channel_scaling), self.max_channels)
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            self.current_channels = out_ch
            self.current_image_dim *= 2
        return layers

    def _make_convolution_layers(self):
        layers = nn.ModuleList()
        while self.current_image_dim > self.min_image_dim:
            in_ch = self.current_channels
            self.current_image_dim //= 2
            out_ch = 1 if self.current_image_dim <= self.min_image_dim else max(in_ch // (2 ** self.channel_scaling), 1)
            if self.pooling:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))

            self.current_channels = out_ch
        return layers

    def forward(self, x):

        x = self.initial_deconv(x)

        for deconv in self.deconvolutions:
            x = deconv(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)
        
        for conv in self.convolutions:
            x = conv(x)
            x = self.leaky_relu(x)
            x = self.dropout(x)

        x = self.downsample(x)

        return x