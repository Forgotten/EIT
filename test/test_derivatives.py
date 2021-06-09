import context
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve

import os
import sys

from eit import *  

# loading the file containing the mesh
mat_fname  = 'data/mesh.mat'
mat_contents = sio.loadmat(mat_fname)

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

# tests for the computation of the gradients 
def u(x,y):
	return np.sin(x)*np.sin(y)

# reference solution for \partial_x u
def dudx(x,y):
	return np.cos(x)*np.sin(y)

# reference solution for \partial_y u
def dudy(x,y):
	return np.sin(x)*np.cos(y)

# sampling at the nodes
xi_u = u(p[:,0], p[:,1])

# computing the necessary matrices
Dx, Dy, M_w = partial_deriv_matrix(v_h)

# change the type of matrix
M_w = spsp.csr_matrix(M_w)

xi_u_x = spsolve(M_w,(Dx@xi_u))
xi_u_y = spsolve(M_w,(Dy@xi_u))

# compute the centroids
centroid = np.sum(np.reshape(p[t,:],(t.shape[0],3,2)), axis = 1)/3;

# compute the reference derivatives
du_dx = dudx(centroid[:,0], centroid[:,1])
du_dy = dudy(centroid[:,0], centroid[:,1])

# compare 
err_x = np.sqrt((xi_u_x-du_dx).T@(M_w@(xi_u_x-du_dx)))/np.sqrt(du_dx.T@(M_w@du_dx))

err_y = np.sqrt((xi_u_y-du_dy).T@(M_w@(xi_u_y-du_dy)))/np.sqrt(du_dy.T@(M_w@du_dy))

# printing the errors
print("Relative error of the derivative in x is %.4e"%err_x)
print("Relative error of the derivative in y is %.4e"%err_y)

