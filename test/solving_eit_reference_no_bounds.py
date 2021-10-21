# Script that computes the reconstruction using the completed 
# DtN map, which has already been computed somewhere else

import context                  # to load the library without installing it 
import scipy.io as sio

# to save the plots witouht conencting to a display
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
from scipy.optimize import Bounds
from scipy.sparse.linalg import spsolve
import time

import scipy.optimize as op

import os
import sys

from eit import *  

# boolean for plotting 
plot_bool = False

# loading the file containing the mesh
# mat_fname  = 'data/adaptive_completion_coarser_DtN_512.mat'
mat_fname  = 'data/completed_new_mesh.mat'
# sigma_fname  = 'data/sigma_vec.mat'

mat_contents = sio.loadmat(mat_fname)
# sigma_contents  = sio.loadmat(sigma_fname)


# points
p = mat_contents['p']
#triangle
t = mat_contents['t']-1 # all the indices should be reduced by one

# volumetric indices
vol_idx = mat_contents['vol_idx'].reshape((-1,))-1 # all the indices should be reduced by one
# indices at the boundaries
bdy_idx = mat_contents['bdy_idx'].reshape((-1,))-1 # all the indices should be reduced by one

# define the mesh
mesh = Mesh(p, t, bdy_idx, vol_idx)

# define the approximation space
v_h = V_h(mesh)

# extracting the DtN data
dtn_data = mat_contents['DtN_i']

# this is the initial guess
sigma_vec_0 = 1 + np.zeros(t.shape[0])

# true perturbation
# sigma_vec = sigma_contents['sigma_vec']
sigma_vec = mat_contents['sigma_vec']


# we create the eit wrapper
eit = EIT(v_h)

# build the stiffness matrices
eit.update_matrices(sigma_vec_0)

# testing the loss and grad 
loss, grad = eit.misfit(dtn_data, sigma_vec_0)

# simple optimization routine
def J(x):
    return eit.misfit(dtn_data, x)

# we define a relatively high tolerance
# recall that this is the square of the misfit
opt_tol = 1e-12

# define a set of the bdy_indices
bdy_idx_set = set(bdy_idx)


bounds = []
for e in range(t.shape[0]):  # integration over one triangular element at a time
    nodes = t[e, :]
    if   (nodes[0] in bdy_idx_set)\
       + (nodes[1] in bdy_idx_set)\
       + (nodes[2] in bdy_idx_set) >= 1:
        bounds.append((1, 1))
    else:
        bounds.append((0, np.inf))


# running the optimization routine
t_i = time.time()
res = op.minimize(J, sigma_vec_0, method='L-BFGS-B',
                   jac = True,
                   tol = opt_tol,
                   bounds=bounds, 
                   options={'maxiter': 10000,
                            'disp': True})
t_f = time.time()
# extracting guess from the resulting optimization 
sigma_guess = res.x

# we show the time it took to run the code
print('Time elapsed in the optimization is %.4f [s]'%(t_f - t_i))

# we proejct sigma back to V in order to plot it
p_v_w = projection_v_w(v_h)
Mass = spsp.csr_matrix(mass_matrix(v_h))

sigma_v = spsolve(Mass, p_v_w@sigma_guess)


# create figure 
plt.figure(figsize=(12,10))
# create a triangulation object 
triangulation = tri.Triangulation(p[:,0], p[:,1], t)
# plot the triangles
# plt.triplot(triangulation, '-k')
# plotting the solution 
plt.tricontourf(triangulation, sigma_v)
# plotting a colorbar
plt.colorbar()
# show
plt.savefig("reference_reconstruction_no_bounds", bbox_inches='tight')   # save the figure to file
#plt.show()


sigma_vec_projected = spsolve(Mass, p_v_w@sigma_vec)

# create figure 
plt.figure(figsize=(12,10))
# create a triangulation object 
triangulation = tri.Triangulation(p[:,0], p[:,1], t)
# plot the triangles
# plt.triplot(triangulation, '-k')
# plotting the solution 
plt.tricontourf(triangulation, sigma_vec_projected)
# plotting a colorbar
plt.colorbar()
# show
plt.savefig("reference_media", bbox_inches='tight')   # save the figure to file
#plt.show()


