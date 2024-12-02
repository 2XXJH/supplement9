import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve

def integrate_linear(m, b, a, c):
    
    def linear_function(x):
        return m * x + b

    result, _ = quad(linear_function, a, c)
    return result























def test_integrate_linear():
    result = integrate_linear(2, 3, 0, 1)  # f(x) = 2x + 3 over [0, 1]
    assert abs(result - 4) < 1e-6, f"Expected 4 but got {result}"

