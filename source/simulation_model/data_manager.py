import meshio
import numpy as np
import os

class DataManager:
    """
    Class to handle the output of the solution.
    """

    def __init__(self, params, points):
        """
        Constructor.

        :param params: Parameters object.
        """

        self.points = points

        # Generate all the points on the grid during initialization
        self.params = params
        self.cells = self.generate_cells(params.nx, params.ny, params.nz)

    def generate_cells(self, nx, ny, nz):
        """
        Generate all the points on the grid.
        :return: Numpy array with the points.
        """

        # Create the cell connectivity (assuming a structured grid with hexahedral elements)
        cells = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    # Define the corners of the hexahedron
                    p1 = k * ny * nx + j * nx + i
                    p2 = p1 + 1
                    p3 = p1 + nx + 1
                    p4 = p1 + nx
                    p5 = p1 + ny * nx
                    p6 = p2 + ny * nx
                    p7 = p3 + ny * nx
                    p8 = p4 + ny * nx
                    # Add the hexahedron (order matters!)
                    cells.append([p1, p2, p3, p4, p5, p6, p7, p8])

        # Convert cells to numpy array and define as meshio format
        cells = [("hexahedron", np.array(cells))]

        return cells


    def save_vtu(self, current_step, T, heat_flux, convective_coefficient = None, error=None, file_name_prefix="output"):
        """
        Output the current solution to the console in 3D.
        """
        
        # Extract flux components for each direction (x, y, z)
        flux_x, flux_y, flux_z = np.transpose(heat_flux)
        
        # Flatten the 3D arrays for each component to match the point data
        flux_x_flat = flux_x.flatten()
        flux_y_flat = flux_y.flatten()
        flux_z_flat = flux_z.flatten()
        
        point_data = {
            "temperature": T.flatten(),  # Flatten temperature array to match points
            "heat_flux_x": flux_x_flat,
            "heat_flux_y": flux_y_flat,
            "heat_flux_z": flux_z_flat
        }

        if convective_coefficient is not None:
            # The convective coefficient is assumed to be a np.array of dim (,nx*ny*nz) flattened using "F" ordering
            point_data["convective_coefficient"] = convective_coefficient

        if error is not None:
            # The error is assumed to be a np.array of dim (,nx*ny*nz) flattened using "F" ordering
            point_data["error"] = error

        # Create the mesh
        mesh = meshio.Mesh(self.points, self.cells, point_data=point_data)
        
        # Get file name       
        output_path = f"{file_name_prefix}_{current_step}.vtu"
        
        # Create folder output_path if it doesn't exist
        if self.params.output_path is not None:
            os.makedirs(self.params.output_path, exist_ok=True)
            output_path = os.path.join(self.params.output_path, output_path)
    
        meshio.write(output_path, mesh)

    def generate_pvd(self, current_step, file_name_prefix='output'):
        """
        write in the pvd at every time step. If the simulation is steady, the time is the current step, otherwise it is the current time.
        """

        if self.params.steady:
            current_time = current_step
        else:
            current_time = f"{current_step * self.params.time_step:.2f}"

        # Write the start of the file if the file does not exist
        file_path = os.path.join(self.params.output_path, "simulation.pvd")
        if not os.path.exists(file_path) or current_step == 0:
            with open(file_path, 'w') as f:
                f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
                f.write('  <Collection>\n')
                f.write('  </Collection>\n')
                f.write('</VTKFile>\n')

        # Read the whole file 
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find where </Collection> is (end of file) and insert vtu file line at that position
        for i, line in enumerate(lines):
            if line.strip() == '</Collection>':
                lines.insert(i, f'    <DataSet timestep="{current_time}" group="" part="0" file="{file_name_prefix}_{current_step}.vtu"/>\n')
                break
        
        # Write the whole file back
        with open(file_path, 'w') as f:
            f.writelines(lines)

    def save_csv(self, params, time_manager, output_path, zone=None,
             temperature=None, PID_output=None, P=None, I=None, D=None,
             Q0=None, T_predicted_avgs=None, Optimized_cost=None):
        """
        Save simulation data to CSV files. 

        Parameters:
        - params: Simulation parameters.
        - time_manager: Time management object.
        - output_path: Directory to save CSV files.
        - zone: Zone index (None for non-zoned systems).
        - temperature: Average temperature of the controlled face.
        - PID_output: PID controller output (convective coefficient).
        - P: Proportional term.
        - I: Integral term.
        - D: Derivative term.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Determine the file name based on whether zones are used
        if zone is not None:
            file_name = f"{output_path}/zone_{zone}_output.csv"
        else:
            file_name = f"{output_path}/output.csv"

        # Build headers dynamically
        headers = ["Time", "Temperature"]
        row = [time_manager.current_time, temperature]

        if params.apply_pid_control:
            headers += ["PID_output", "P", "I", "D"]
            row += [PID_output, P, I, D]

        if params.apply_mpc_control:
            headers += ["Q0", "T_predicted_avgs", "Optimized_cost"]
            row += [Q0, T_predicted_avgs, Optimized_cost]

        # Cleaning
        def clean_value(value):
            if isinstance(value, (list, np.ndarray)):
                return ";".join(str(float(x)) for x in value)
            elif isinstance(value, np.generic):
                return str(float(value))
            else:
                return str(value)

        row_str = [clean_value(v) for v in row]

        # If file doesn’t exist, create header
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                f.write(",".join(headers) + "\n")

        # Append the current simulation data to the file
        with open(file_name, 'a') as f:
            f.write(",".join(map(str, row)) + "\n")

    def save_adjoint_output(self, iteration, error, convective_coefficient, output_folder):

        file_name = f"{output_folder}/adjoint.csv"
        h_conv_file_name = f"{output_folder}/adjoint_h_conv.txt"
        
        # If the file doesn't exist, create it with headers
        if not os.path.exists(file_name):
            with open(file_name, 'w') as f:
                f.write("iteration, error\n")

        else:
            with open(file_name, 'a') as f:
                f.write(f"{iteration}, {error}\n")


        with open(h_conv_file_name, 'w') as f:
            f.write("convective_coefficient\n")
            f.write(f"{convective_coefficient}\n")


