'''
Copyright 2024-2025, the Active Cooling Experimental Application Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np

class PIDControl:
    def __init__(self):
        self.previous_error = 0
        self.gains = np.zeros(3)
        self.output = 0
        self.integral_error = 0

    def compute_output(self, current_temperature, setpoint, time_step, current_flow_rate, flow_rate_saturation_min = 5, flow_rate_saturation_max = 300, output_min = 0, output_max = 300):
        Kp = self.gains[0]
        Ki = self.gains[1]
        Kd = self.gains[2]

        if setpoint is None or isinstance(setpoint, int):
            error = 0
        elif setpoint < 1000:
            error = current_temperature - setpoint
            
        else:
            error = 0

        # Update integral errors (only update if not saturated)
        if current_flow_rate <= flow_rate_saturation_min or current_flow_rate >= flow_rate_saturation_max:
            self.integral_error = self.integral_error
        else:
            self.integral_error += error*time_step

        # Update derivative
        derivative = (error - self.previous_error) / time_step

        self.output = Kp * error + Ki * self.integral_error + Kd * derivative 
        self.previous_error = error

        self.output = min(output_max, max(output_min, self.output))

        return self.output