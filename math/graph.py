import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

L = np.linspace(0, 300, 30)
K = np.linspace(0, 300, 30)
L, K = np.meshgrid(L, K)

P = 1.01 * (L**0.75) * (K**0.25)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(L, K, P, cmap='viridis')

fig.colorbar(surface)

ax.view_init(elev=15, azim=225)

ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

ax.set_xlabel('L-axis')
ax.set_ylabel('K-axis')
ax.set_zlabel('P-axis')

plt.show()
