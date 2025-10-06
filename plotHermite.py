import numpy as np
import matplotlib.pyplot as plt

def E_of_T(T, T0, T1, E0, E1, theta0_deg, theta1_deg):
    """Cubic Hermite interpolation for Young's modulus E(T).
       Slopes specified as angles in degrees."""
    # Convert slope angles (degrees) to slope values
    m0 = np.tan(np.radians(theta0_deg))
    m1 = np.tan(np.radians(theta1_deg))
    print(f"Slopes: m0={m0}, m1={m1}")
    
    # Normalize to [0,1]
    t = (T - T0) / (T1 - T0)
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    return h00*E0 + h10*(T1-T0)*m0 + h01*E1 + h11*(T1-T0)*m1

# Example

# We decided on this
T0, T1 = 0, 1250


for example in [1, 2]:
    if (example == 1):    # This is a sample for 17-4PH SS
        E0, E1 = 196e9, 8e10     # Young's modulus at endpoints
        theta0, theta1 = -3, -0.5   # slopes given as angles (degrees)
        material = "17-4PH Stainless Steel"
    else:     # This is a sample for 7078Al
        E0, E1 = 7.3e10, -7e10     # Young's modulus at endpoints
        theta0, theta1 = -40, 0  # slopes given as angles (degrees)
        material = "7078 Aluminum"
    T_vals = np.linspace(T0, T1, 300)
    E_vals = E_of_T(T_vals, T0, T1, E0, E1, theta0, theta1)

    plt.plot(T_vals, E_vals, label=material)
plt.xlabel("Temperature (T)")
plt.ylabel("Young's Modulus (E)")
plt.legend()
plt.grid(True)
plt.show()
