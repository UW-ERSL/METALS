import numpy as np
import torch
import pandas as pd

TMin = 0
TMax = 1200

T0_CUBIC = 0
T1_CUBIC = 400
T2_CUBIC = 800
T3_CUBIC = 1200

T0_BEZIER = 0
T1_BEZIER = 400
T2_BEZIER = 800
T3_BEZIER = 1200

def cubicInterpolation(T, M0, M1, M2, M3, T0=T0_CUBIC, T1=T1_CUBIC, T2=T2_CUBIC, T3=T3_CUBIC):
    """Cubic interpolation for material property variation with temperature using four points."""
    # Solve for cubic coefficients: f(T) = a*T^3 + b*T^2 + c*T + d
    A = np.array([
        [T0**3, T0**2, T0, 1],
        [T1**3, T1**2, T1, 1],
        [T2**3, T2**2, T2, 1],
        [T3**3, T3**2, T3, 1]
    ])
    b = np.array([M0, M1, M2, M3])
    coeffs = np.linalg.solve(A, b)
    a, b_, c, d = coeffs
    return a*T**3 + b_*T**2 + c*T + d


def cubicInterpolation_torch(T, M0, M1, M2, M3, T0=T0_CUBIC, T1=T1_CUBIC, T2=T2_CUBIC, T3=T3_CUBIC):
    """Cubic interpolation for material property variation with temperature using four points (torch version)."""
    # Solve for cubic coefficients: f(T) = a*T^3 + b*T^2 + c*T + d
    A = torch.tensor([
        [T0**3, T0**2, T0, 1],
        [T1**3, T1**2, T1, 1],
        [T2**3, T2**2, T2, 1],
        [T3**3, T3**2, T3, 1]
    ], dtype=torch.float32)
    b = torch.tensor([M0, M1, M2, M3], dtype=torch.float32)
    coeffs = torch.linalg.solve(A, b)
    a, b_, c, d = coeffs
    return a*T**3 + b_*T**2 + c*T + d

def logBezierInterpolation(T, M0, M1, M2, M3, T0=TMin,  T3=TMax):
    """Cubic Bezier interpolation in log10-space for material property variation with temperature using four points."""
    # Take log10 of all M values (handle zeros or negatives gracefully)
    M0_log = np.log10(M0)
    M1_log = np.log10(M1)
    M2_log = np.log10(M2)
    M3_log = np.log10(M3)
    # Normalize T to [0,1] using T0 and T3
    t = (T - T0) / (T3 - T0)
    # Cubic Bezier basis functions
    B0 = (1 - t) ** 3
    B1 = 3 * (1 - t) ** 2 * t
    B2 = 3 * (1 - t) * t ** 2
    B3 = t ** 3
    log_val = B0 * M0_log + B1 * M1_log + B2 * M2_log + B3 * M3_log
    return 10 ** log_val


def logBezierInterpolation_torch(T, M0, M1, M2, M3, T0=TMin,  T3=TMax):
    """Cubic Bezier interpolation in log10-space for material property variation with temperature using four points (torch version)."""
    M0_log = torch.log10(M0)
    M1_log = torch.log10(M1)
    M2_log = torch.log10(M2)
    M3_log = torch.log10(M3)
    t = torch.tensor((T - T0) / (T3 - T0))
    B0 = (1 - t) ** 3
    B1 = 3 * (1 - t) ** 2 * t
    B2 = 3 * (1 - t) * t ** 2
    B3 = t ** 3
    log_val = B0 * M0_log + B1 * M1_log + B2 * M2_log + B3 * M3_log
    return 10 ** log_val

def bezierInterpolation(T, M0, M1, M2, M3, T0=T0_BEZIER, T1=T1_BEZIER, T2=T2_BEZIER, T3=T3_BEZIER):
    """Cubic Bezier interpolation for material property variation with temperature using four points."""
    # Normalize T to [0,1] using T0 and T3
    t = (T - T0) / (T3 - T0)
    # Cubic Bezier basis functions
    B0 = (1 - t) ** 3
    B1 = 3 * (1 - t) ** 2 * t
    B2 = 3 * (1 - t) * t ** 2
    B3 = t ** 3
    return B0 * M0 + B1 * M1 + B2 * M2 + B3 * M3

def bezierInterpolation_torch(T, M0, M1, M2, M3, T0=T0_BEZIER, T1=T1_BEZIER, T2=T2_BEZIER, T3=T3_BEZIER):
    """Cubic Bezier interpolation for material property variation with temperature using four points (torch version)."""
    # Normalize T to [0,1] using T0 and T3
    t = (T - T0) / (T3 - T0)
    # Cubic Bezier basis functions
    B0 = (1 - t) ** 3
    B1 = 3 * (1 - t) ** 2 * t
    B2 = 3 * (1 - t) * t ** 2
    B3 = t ** 3
    return B0 * M0 + B1 * M1 + B2 * M2 + B3 * M3


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
    Data = [7.30E+10, 4.00E+9, 2.00E+06, 2.00E+03]

    # Plot Bezier interpolation using the Data array
    T_vals = np.linspace(TMin, TMax, 300)
    vals = logBezierInterpolation(T_vals, *Data)
    plt.semilogy(T_vals, vals, linestyle='--')
    plt.grid(True)
    plt.show()


    # for example in [1, 2]:
    #     if (example == 1):    # This is a sample for 17-4PH SS
    #         E0, E1 = 196e9, 8e10     # Young's modulus at endpoints
    #         theta0, theta1 = -3, -0.5   # slopes given as angles (degrees)
    #         material = "17-4PH Stainless Steel"
    #     else:     # This is a sample for 7078Al
    #         E0, E1 = 7.3e10, -7e10     # Young's modulus at endpoints
    #         theta0, theta1 = -40, 0  # slopes given as angles (degrees)
    #         material = "7078 Aluminum"
    #     T_vals = np.linspace(TMin, TMax, 300)
    #     E_vals = hermiteInterpolation(T_vals,  E0, E1, theta0, theta1)

    #     plt.plot(T_vals, E_vals, label=material)
    # plt.xlabel("Temperature (T)")
    # plt.ylabel("Young's Modulus (E)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
