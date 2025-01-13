'''
Copyright 2024-2025, the Active Cooling Experimental Application Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

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
            
        self.max = 120
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
    
