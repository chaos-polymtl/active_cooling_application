import numpy as np

class Temperature():
    def __init__(self, n_region, test = False):
        self.test = test

        if self.test:
            pass
        else:
            from source.thermal_cam import ThermalCam
            self.thermal_cam = ThermalCam()

        # Tuple of the resolution of the camera
        self.resolution = (24,32)

        # Emissivity
        self.emissivity = 0.95
            
        self.max = 100
        self.min = 30

        self.temperature = np.zeros(self.resolution[0] * self.resolution[1])
        self.temperature_grid = self.temperature.reshape(self.resolution[0], self.resolution[1])

        # Container for temperature average per region
        self.temperature_average = np.zeros(n_region)

    def get_temperature(self):
        if self.test:
            self.temperature = np.genfromtxt('source/test_data/temperature_static.csv', delimiter = ",", dtype=np.float32)[1:]
        else:
            self.thermal_cam.get_temperature()
            self.temperature = self.thermal_cam.temperature

        self.temperature_grid = self.temperature.reshape(self.resolution[0], self.resolution[1])

    def get_temperature_average(self, n_region, region_boundaries):
            '''Get temperature average within regions'''

            # Get temperature average within regions
            for i in range(n_region):

                # Get region boundaries
                y_min, y_max, x_min, x_max = region_boundaries[i]
            
                # Calculate average temperature within patched region
                self.temperature_average[i] = np.mean(self.temperature_grid[x_min:x_max, y_min:y_max])
    
