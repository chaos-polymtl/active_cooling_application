import numpy as np

class PIDControl:
    def __init__(self):
        self.previouhs_error = 0
        self.gains = np.zeros(3)
        self.output = 0
        self.integral_error = 0

    def PID(self, current_temperature, setpoint, time_step):
        Kp = self.gains[0]
        Ki = self.gains[1]
        Kd = self.gains[2]

        error = current_temperature - setpoint
        error[setpoint <= 0] = 0

        # Update integral errors
        self.integral_errors = np.roll(self.integral_errors, shift=-1, axis=1)
        self.integral_errors[:, -1] = error

    
        integral = self.integral_errors.sum()
        derivative = (error - self.previous_error) / time_step
        
        self.output = error * Kp + Ki * integral * time_step + Kd * derivative
        self.previous_error = error

        return self.output