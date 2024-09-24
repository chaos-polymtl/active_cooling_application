# Module to work with infra-red camera MLX 90640 using Raspberry Pi

# Import libraries
import numpy as np
import busio
import board
import adafruit_mlx90640 as amlx

# Define thermal camera class
class thermal_cam:

    # Default constructor
    def __init__(self, test = False):

        self.test = test
        
        # Tuple of the resolution of the camera
        self.resolution = (24,32)

        # Emissivity
        self.emissivity = 0.95

        # Setup I2C connection with camera
        self.i2c_connection = busio.I2C(board.SCL, board.SDA)

        # Start library
        self.mlx = amlx.MLX90640(self.i2c_connection)

        # Set refresh rate
        self.mlx.refresh_rate = amlx.RefreshRate.REFRESH_1_HZ

        # Temperature vector
        self.temperature = np.zeros(self.resolution[0]*self.resolution[1])

        # Add default max and min temperature
        self.max = 100
        self.min = 30
        self.temperature_grid = self.temperature.reshape(self.resolution[0], self.resolution[1])

    def get_temperature(self):
        if self.test:
            self.temperature = np.genfromtxt('source/test_data/temperature_static.csv', delimiter = ",", dtype=np.float32)[1:]
        else:
            self.getFrame(self.temperature)
            
        self.temperature = np.round(self.temperature, decimals=2)
        self.temperature_grid = self.temperature.reshape(self.resolution[0], self.resolution[1])


    def getFrame(self, framebuf):
        OPENAIR_TA_SHIFT = 8
        emissivity = self.emissivity
        tr = 23.15
        mlx90640Frame = [0] * 834
        status = self.mlx._GetFrameData(mlx90640Frame)

        if status < 0:
            pass
        else:
            # For a MLX90640 in the open air the shift is -8 degC.
            tr = self.mlx._GetTa(mlx90640Frame) - OPENAIR_TA_SHIFT
            self.mlx._CalculateTo(mlx90640Frame, emissivity, tr, framebuf)


    # Temperature setter
    def set_range(self, min, max):        
        # Set temperature
        self.min = min
        self.max = max

    # File path setter
    def set_csv_path(self, new_file_path):

        # New file path
        self.output_csv = new_file_path

    # Image folder path setter
    def set_image_folder(self, new_folder_path):

        # New folder path
        self.output_images_path = new_folder_path
            
    

    

