import numpy as np
import torch
import pandas as pd

TMin = 0
TMax = 1250
def hermiteInterpolation(T, M0, M1, theta0_deg, theta1_deg, T0 = TMin, T1 = TMax): # Assume data between T0 and T1
    """Cubic Hermite interpolation for material property variation with temperature.
       Slopes specified as angles in degrees."""
    # Convert slope angles (degrees) to slope values
    m0 = np.tan(np.radians(theta0_deg))
    m1 = np.tan(np.radians(theta1_deg))
   
    # Normalize to [0,1]
    t = (T - T0) / (T1 - T0)

    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
   
    return h00*M0 + h10*(T1-T0)*m0 + h01*M1 + h11*(T1-T0)*m1

def hermiteInterpolation_torch(T, M0, M1, theta0_deg, theta1_deg, T0 = TMin, T1 = TMax):
    """Cubic Hermite interpolation for material property variation with temperature.
       Slopes specified as angles in degrees. All inputs except T are torch tensors."""
    # Convert slope angles (degrees) to slope values
    m0 = torch.tan(torch.deg2rad(theta0_deg))
    m1 = torch.tan(torch.deg2rad(theta1_deg))

    # Normalize to [0,1]
    t = (T - T0) / (T1 - T0)
    h00 = torch.tensor(2*t**3 - 3*t**2 + 1)
    h10 = torch.tensor(t**3 - 2*t**2 + t)
    h01 = torch.tensor(-2*t**3 + 3*t**2)
    h11 = torch.tensor(t**3 - t**2)
    return h00*M0 + h10*(T1-T0)*m0 + h01*M1 + h11*(T1-T0)*m1

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # We decided on this
    

    for example in [1, 2]:
        if (example == 1):    # This is a sample for 17-4PH SS
            E0, E1 = 196e9, 8e10     # Young's modulus at endpoints
            theta0, theta1 = -3, -0.5   # slopes given as angles (degrees)
            material = "17-4PH Stainless Steel"
        else:     # This is a sample for 7078Al
            E0, E1 = 7.3e10, -7e10     # Young's modulus at endpoints
            theta0, theta1 = -40, 0  # slopes given as angles (degrees)
            material = "7078 Aluminum"
        T_vals = np.linspace(TMin, TMax, 300)
        E_vals = hermiteInterpolation(T_vals,  E0, E1, theta0, theta1)

        plt.plot(T_vals, E_vals, label=material)
    plt.xlabel("Temperature (T)")
    plt.ylabel("Young's Modulus (E)")
    plt.legend()
    plt.grid(True)
    plt.show()
