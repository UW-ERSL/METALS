import pandas as pd

# ---- Data ----
data = [
    ["17-4PH SS", 7850, 2.10E11,1.855E11,1.610E11,1.365E11,
     2.50E08,2.083E08,1.667E08,1.25E08,
     1.12E-05,1.28E-05,1.44E-05,1.60E-05,
     2.26E01,2.38E01,2.50E01,2.62E01],

    ["Grade 304 SS", 7800, 2.00E11,1.733E11,1.467E11,1.20E11,
     1.80E08,1.44E08,1.08E08,3.60E07,
     1.84E-05,1.90E-05,1.97E-05,2.03E-05,
     2.15E01,2.28E01,2.40E01,2.52E01],

    ["7068 Al", 2700, 7.00E10,5.483E10,3.967E10,2.45E10,
     1.00E08,7.00E07,4.00E07,1.00E07,
     2.40E-05,2.60E-05,2.80E-05,3.00E-05,
     2.36E02,2.12E02,1.88E02,1.64E02]
]

columns = [
    "Material","Density",
    "E0","E1","E2","E3",
    "Y0","Y1","Y2","Y3",
    "Alpha0","Alpha1","Alpha2","Alpha3",
    "K0","K1","K2","K3"
]

# ---- Create DataFrame ----
df = pd.DataFrame(data, columns=columns)

# ---- Export to Excel ----
df.to_excel("material_properties.xlsx", index=False)