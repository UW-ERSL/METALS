import numpy as np

import matplotlib.pyplot as plt

# Real materials points
real_materials = np.array([
    [-1.5, 1.0],
    [1.5, 1.5],
    [-1.0, -1.0],
    [0.5, -0.65]
])

# Latent point Z
latent_point = np.array([0, 0])

plt.figure(figsize=(6,6))
plt.grid(True)

# Plot real materials
plt.scatter(real_materials[:,0], real_materials[:,1], marker='*', color='black', label='Real Materials')

# Plot latent point Z
plt.scatter(latent_point[0], latent_point[1], color='red', s=200, label='Latent Point Z')

# Draw dashed lines from latent point to real materials
for pt in real_materials:
    plt.plot([latent_point[0], pt[0]], [latent_point[1], pt[1]], 'k--', linewidth=1)

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel('$z_0$', fontsize=18)
plt.ylabel('$z_1$', fontsize=18)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()