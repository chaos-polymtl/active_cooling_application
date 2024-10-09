import numpy as np

class PIDControl:
    def __init__(self):
        self.previous_error = 0
        self.gains = np.zeros(3)
        self.output = 0
        self.integral_error = 0

    def compute_output(self, current_temperature, setpoint, time_step, current_flow_rate, flow_rate_saturation_min = 5, flow_rate_saturation_max = 100, output_min = 0, output_max = 100):
        Kp = self.gains[0]
        Ki = self.gains[1]
        Kd = self.gains[2]

        if setpoint < 1000:
            error = current_temperature - setpoint
        else:
            error = 0

        # Update integral errors
        # Only update if not saturated
        if current_flow_rate <= flow_rate_saturation_min or current_flow_rate >= flow_rate_saturation_max:
            self.integral_error = self.integral_error
            
        else:
            self.integral_error += error

        # Update derivative
        derivative = (error - self.previous_error) / time_step

        self.output = error * Kp + Ki * self.integral_error * time_step + Kd * derivative
        self.previous_error = error

        self.output = min(output_max, max(output_min, self.output))

        return self.output
