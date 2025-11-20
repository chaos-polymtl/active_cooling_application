class TimeManager:
    def __init__(self, params):
        """
        Initialize the TimeManager.

        :param start_time: Initial simulation time (default: 0.0)
        :param time_step: Time step size (default: 0.01)
        :param final_time: Optional final simulation time (default: None)
        """
        self.current_time = params.start_time
        self.time_step = params.time_step
        self.final_time = params.final_time
        self.current_step = 0

    def get_time(self):
        """
        Get the current simulation time.
        
        :return: The current time
        """
        return self.current_time

    def update_time(self):
        """
        Advance the current time by the time step.
        
        :return: The updated time
        """
        self.current_time += self.time_step
        self.current_step += 1
        return self.current_time

    def is_finished(self):
        """
        Check if the final time has been reached (if a final time is set).
        
        :return: True if the final time has been reached, False otherwise
        """
        if self.final_time is not None:
            return (self.current_time + 1e-8) > (self.final_time)
        return False

