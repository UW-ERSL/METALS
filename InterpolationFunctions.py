import numpy as np
import torch
import pandas as pd

# These temperature values are used in several interpolation functions
TMin = 0
TMax = 1200

T0_CUBIC = TMin
T1_CUBIC = TMin + (TMax - TMin) / 3
T2_CUBIC = TMin + 2 * (TMax - TMin) / 3
T3_CUBIC = TMax

T0_BEZIER = TMin
T1_BEZIER = TMin + (TMax - TMin) / 3
T2_BEZIER = TMin + 2 * (TMax - TMin) / 3
T3_BEZIER = TMax

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

def findControlPointsLogBezier(T_data, M_data, T0=TMin, T3=TMax):
    """Find control points for log-space Bezier interpolation given data points."""
    # Take log10 of M_data
    M_log = np.log10(M_data)
    # Normalize T_data to [0,1]
    t_data = (T_data - T0) / (T3 - T0)
    n = len(T_data)
    if n < 4:
        raise ValueError("At least four data points are required to determine control points.")
    
    # Set up the system of equations
    A = np.zeros((n, 4))
    for i in range(n):
        t = t_data[i]
        A[i, 0] = (1 - t) ** 3
        A[i, 1] = 3 * (1 - t) ** 2 * t
        A[i, 2] = 3 * (1 - t) * t ** 2
        A[i, 3] = t ** 3
    
    # Solve the least squares problem to find control points
    control_points_log, _, _, _ = np.linalg.lstsq(A, M_log, rcond=None)
    control_points = 10 ** control_points_log  # Convert back from log space
    return control_points   


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example of Bezier interpolation
    # Data points for interpolation
    K = [20, 19, 17.6, 15.5]  # Thermal conductivity data (sample values)
    # Plot Bezier interpolation using the Data array
    T_vals = np.linspace(TMin, TMax, 300)
    vals = bezierInterpolation(T_vals, *K)   
    plt.plot(T_vals, vals, linestyle='-')
    # Plot control points for Bezier interpolation
    control_T = [T0_BEZIER, T1_BEZIER, T2_BEZIER, T3_BEZIER]
    plt.plot(control_T, K, 'r*', label="Control Points")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Conductivity K (W/m-K)")
    plt.show()

    # Example of Log Bezier interpolation for Elastic Modulus
    # Data points for interpolation
    E_steel = [1.9e11, 1.1e11, 9e10, 8e10]  # Elastic modulus data (sample values)
    # Plot Log Bezier interpolation using the Data array
    T_vals = np.linspace(TMin, TMax, 300)
    vals = logBezierInterpolation(T_vals, *E_steel)
    plt.semilogy(T_vals, vals, linestyle='--')
    # Plot control points for Log Bezier interpolation
    control_T = [TMin, TMax / 3, 2 * TMax / 3, TMax]
    plt.semilogy(control_T, E_steel, 'g*', label="Control Points")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Youngs Modulus Steel (Pa)")
    plt.show()


    E_Al = [7.3e10, 4e10, 2e7, 2e6]  # Elastic modulus data (sample values)
    # Plot Log Bezier interpolation using the Data array
    T_vals = np.linspace(TMin, TMax, 300)
    vals = logBezierInterpolation(T_vals, *E_Al)
    plt.plot(T_vals, vals, linestyle='--')
    # Plot control points for Log Bezier interpolation
    control_T = [TMin, TMax / 3, 2 * TMax / 3, TMax]
    plt.plot(control_T, E_Al, 'g*', label="Control Points")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Youngs Modulus Aluminum (Pa)")
    plt.show()


    # Example of finding control points for Log Bezier interpolation
    # Sample data points (Temperature vs Elastic Modulus)
    T_data = np.array([25, 100, 125, 150, 200, 250, 300])  # Sample temperature data
    M_data = 1e9 * np.array([70, 68, 66, 64, 60, 52, 40])  # Sample elastic modulus data
    
    control_points = findControlPointsLogBezier(T_data, M_data)
    print("Control points for Log Bezier interpolation:", control_points)
    # Verify by plotting
    T_vals = np.linspace(TMin, TMax, 300)
    M_vals = logBezierInterpolation(T_vals, *control_points)    
    plt.semilogy(T_vals, M_vals, label="Log Bezier Interpolation")
    plt.semilogy(T_data, M_data, 'ro', label="Data Points")       
    plt.semilogy([TMin, TMax / 3, 2 * TMax / 3, TMax], control_points, 'g*', label="Control Points")
    plt.legend()
    plt.grid(True)  
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Elastic Modulus (Pa)")
    plt.show()