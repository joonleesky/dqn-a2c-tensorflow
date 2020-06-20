class LinearScheduler(object):
    
    def __init__(self, initial_value, final_value, schedule_timesteps):
        """
        Linear Interpolation between initial_value to the final_value
        
        Parameters
        ----------
        initial_value: float
            initial output value
        final_value: float
            final output value
        schedule_timesteps: int
            number of timesteps to lineary anneal initial value to the final value
        """
        self.initial_value = initial_value
        self.final_value   = final_value
        self.schedule_timesteps = schedule_timesteps
        
    def value(self, t):
        """
        Return the scheduled value at time step t
        
        Parameters
        ----------
        t: int
            time steps
        """
        interval = (self.initial_value - self.final_value) / self.schedule_timesteps
        # After the schedule_timesteps, final value is returned
        return max(self.initial_value - interval * t, self.final_value)
    
class StairCaseScheduler(object):
    
    def __init__(self, initial_value, final_value, schedule_timesteps):
        """
        Scheduler based on the Stair-Case function
        Whenever the timestep hit the value is changed to the final_value
        
        Parameters
        ----------
        initial_value: float
            initial output value
        final_value: float
            final output value
        schedule_timesteps: int
            number of timesteps to drop initial value to the final value
        """
        self.initial_value = initial_value
        self.final_value   = final_value
        self.schedule_timesteps = schedule_timesteps
        
    def value(self, t):
        """
        Return the scheduled value at time step t
        
        Parameters
        ----------
        t: int
            time steps
        """
        if t < self.schedule_timesteps:
            return self.initial_value
        else:
            return self.final_value

class ConstantScheduler(object):
    
    def __init__(self, initial_value):
        """
        Linear Interpolation between initial_value to the final_value
        
        Parameters
        ----------
        value: float
            output value
        """
        self.initial_value = initial_value
        
    def value(self, t):
        """
        Return the scheduled value at time step t
        
        Parameters
        ----------
        t: int
            time steps
        """
        return self.initial_value