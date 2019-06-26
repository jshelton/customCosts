import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np


# Theta
theta_min, theta_max = 1, 30
xmin, xmax = theta_min, theta_max

# Z
Zmin, Zmax = 1, 100
ymin, ymax = Zmin, Zmax


# make these smaller to increase the resolution
dx, dy = 0.1, 0.1


# generate 2 2d grids for the x & y bounds
Y, X = np.mgrid[slice(ymin, ymax + dy, dy),
                slice(xmin, xmax + dx, dx)]


# c = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
c = np.zeros((len(Y), len(Y[0])))

aik = np.zeros(len(Y))


q1 = 1.5
q2 = 3
Q1 = 5
Q2 = 5
p1 = 2
p2 = 3
lambdaL = 25

for i in range(len(Y)):
    aik[i] = (Y[i][0]**2/2)**(1/3)
    A = aik[i]
    I = aik[i]
    K = aik[i]

    for j in range(len(Y[0])):
        x = X[i][j]
        y = Y[i][j]
        # if ((x/5)**2 > (y/5)**3):
        #     c[i][j] = 2
        # if (x + y < 4):
        #     c[i][j] = 1
        if (Y[i][0] < theta_min*lambdaL**.5):
            c[i][j] = 1


# x and y are bounds, so c should be the value *inside* those bounds.
# Therefore, remove the last value from the c array.
c = c[:-1, :-1]
levels = MaxNLocator(nbins=15).tick_values(c.min(), c.max())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)


# fig, (ax1, ax0) = plt.subplots(nrows=1)
fig, (ax1) = plt.subplots()

# im = ax0.pcolormesh(X, Y, c, cmap=cmap, norm=norm)
# fig.colorbar(im, ax=ax0)
# ax0.set_title('pcolormesh with levels')


# contours are *point* based plots, so convert our bound into point
# centers
cf = ax1.contourf(X[:-1, :-1] + dx/2.,
                  Y[:-1, :-1] + dy/2., c, levels=levels,
                  cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title('contourf with levels')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

plt.show()
