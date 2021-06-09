
# importing this file so we don't need to install the package
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

# loading the file
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

# defining the conductivity
def sigma(x,y):
	return  1 + 0*x + 0*y

#compute the centroids at each triangle
centroid = np.sum(np.reshape(p[t,:],(t.shape[0],3,2)), axis = 1)/3;

# evaluate sigma at each centroid
sigma_vec = sigma(centroid[:,0], centroid[:,1])

# build the stiffness matrix
S = stiffness_matrix(v_h, sigma_vec)

# reduced Stiffness matrix (only volumetric dof)
Sb = spsp.csr_matrix(S[vol_idx,:][:,vol_idx])

# building the boundary conditions
def bdy_cond(x,y):
	return 1+x+y

bdy_points = p[bdy_idx,:].reshape((-1,2))
bdy_data = bdy_cond(bdy_points[:,0], bdy_points[:,1])

# building the rhs for the linear system
Fb = -S[vol_idx,:][:,bdy_idx]*bdy_data

# solve interior dof
U_vol = spsolve(Sb, Fb)

# allocate the space for the full solution
sol = np.zeros((p.shape[0],))

# write the corresponding values back to the solution
sol[bdy_idx] = bdy_data
sol[vol_idx] = U_vol

# plotting the solution 
# create a triangulation object 
triangulation = tri.Triangulation(p[:,0], p[:,1], t)
# plot the triangles
plt.triplot(triangulation, '-k')
# plotting the solution 
plt.tricontourf(triangulation, sol)
# plotting a colorbar
plt.colorbar()
# show
plt.show()

err = np.sqrt(np.sum(np.square(bdy_cond(p[:,0], p[:,1])-sol)))

print("Error of the approximation is %.4e"%err)

