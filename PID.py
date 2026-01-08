class Proportional:
    def __init__(self, Kp):
        self.Kp = Kp

    def calc(self, error):
        return self.Kp * error

class Integral:
    def __init__(self, Ki):
        self.Ki = Ki
        self.integral = 0
    def calc(self, error, dt):
        self.integral += error * dt
        return self.Ki * self.integral

class Derivative:
    def __init__(self, Kd):
        self.Kd = Kd
        self.prev_error = 0

    def calc(self, error, dt):
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kd * derivative
class PIDsim:
    def __init__(self, Kp, Ki, Kd, dt= 0.1):
        self.P = Proportional(Kp)
        self.I = Integral(Ki)
        self.D = Derivative(Kd)
        self.dt = dt
        self.prev_time = 0

    def update(self, setpoint, current_value):
        
        error = setpoint - current_value
    
        P_t = self.P.calc(error)
        I_t = self.I.calc(error, self.dt)
        D_t = self.D.calc(error, self.dt)
        
        self.last_time += self.dt
        
        return P_t + I_t + D_t
