import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np


# Theta (X axis)
theta_min, theta_max = 1, 30
Xmin, Xmax = theta_min, theta_max

# Z (Y axis)
Zmin, Zmax = 1, 30
Ymin, Ymax = Zmin, Zmax


# make these smaller to increase the resolution
dx, dy = 0.1, 0.1


# generate 2 2d grids for the X & Y bounds
Y_mat, X_mat = np.mgrid[slice(Ymin, Ymax + dy, dy),
                        slice(Xmin, Xmax + dx, dx)]


# c = np.sin(X)**10 + np.cos(10 + Y*X) * np.cos(X)
c = np.zeros((len(Y_mat), len(Y_mat[0])))

aik_arr = np.zeros(len(Y_mat))


q1 = 1.5
q2 = 3
Q1 = 5
Q2 = 5
p1 = 2
p2 = 3
Lambdas = [40, 40, 40]
# Assumes that additional products are added at the end and not in the middle

for i in range(len(Y_mat)):
    Z = Y_mat[i][0]

    aik = (Z**2/2)**(1/3)

    aik_arr[i] = aik
    A = aik
    I = aik
    K = aik

    for j in range(len(Y_mat[0])):
        X = X_mat[i][j]
        Y = Y_mat[i][j]

        Theta = X

        # if ((X/5)**2 > (Y/5)**3):
        #     c[i][j] = 2
        # if (X + Y < 4):
        #     c[i][j] = 1

        if (Y_mat[i][0] < theta_min*Lambdas[0]**.5):
            c[i][j] = 1

        # Case i - two products

        # sum = 0
        # for k in range(2):
        #     sum += (Lambdas[k]**.5 - Z)**2

        # prodi = sum/(4*A)


# X and Y are bounds, so c should be the value *inside* those bounds.
# Therefore, remove the last value from the c array.
c = c[:-1, :-1]
levels = MaxNLocator(nbins=15).tick_values(c.min(), c.max())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)


fig, (ax1) = plt.subplots()

# If you wanted to plot two images
# fig, (ax1, ax0) = plt.subplots(nrows=1)
# im = ax0.pcolormesh(X_mat, Y_mat, c, cmap=cmap, norm=norm)
# fig.colorbar(im, ax=ax0)
# ax0.set_title('pcolormesh with levels')


# contours are *point* based plots, so convert our bound into point
# centers
cf = ax1.contourf(X_mat[:-1, :-1] + dx/2.,
                  Y_mat[:-1, :-1] + dy/2., c, levels=levels,
                  cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title('contourf with levels')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

plt.show()
