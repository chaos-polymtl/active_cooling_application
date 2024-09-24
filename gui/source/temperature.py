import numpy as np

class Temperature():
    def __init__(self, test = False):
        self.test = test

        if self.test:
            pass
        else:
            from .thermal_cam import ThermalCam

        # Tuple of the resolution of the camera
        self.resolution = (24,32)

        # Emissivity
        self.emissivity = 0.95
            
        self.max = 100
        self.min = 30

        self.temperature = np.zeros(self.resolution[0] * self.resolution[1])
        self.temperature_grid = self.temperature.reshape(self.resolution[0], self.resolution[1])

    def get_temperature(self):
        if self.test:
            self.temperature = np.genfromtxt('source/test_data/temperature_static.csv', delimiter = ",", dtype=np.float32)[1:]
        else:
            self.measure_temperature()

        self.temperature_grid = self.temperature.reshape(self.resolution[0], self.resolution[1])
