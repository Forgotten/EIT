import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
from pypardiso import spsolve
import scipy.integrate as integrate

import numba
from numba import jit, void, int64, float64, uint16

from .fem import Mesh, V_h, partial_deriv_matrix, mass_matrix

class EIT:
    def __init__(self, v_h):
        self.v_h = v_h
        self.build_matrices()

    def update_matrices(self, sigma_vec):

        vol_idx = self.v_h.mesh.vol_idx
        bdy_idx = self.v_h.mesh.bdy_idx

        S = stiffness_matrix_numba(self.v_h, sigma_vec)
        self.S  = spsp.csr_matrix(S)
        self.S_ii = spsp.csr_matrix(self.S[vol_idx,:][:,vol_idx])
        self.S_ib = spsp.csr_matrix(self.S[vol_idx,:][:,bdy_idx])

    def build_matrices(self):

        self.Mass = mass_matrix(self.v_h)
        Kx, Ky, M_w = partial_deriv_matrix(self.v_h)

        self.Dx = spsp.diags(1/M_w.diagonal())@Kx
        self.Dy = spsp.diags(1/M_w.diagonal())@Ky
        self.M_w = M_w

    def dtn_map(self, sigma_vec):
        # do this here

        self.update_matrices(sigma_vec)

        n_bdy_pts = len(self.v_h.mesh.bdy_idx)
        n_pts  = self.v_h.mesh.p.shape[0]
    
        vol_idx = self.v_h.mesh.vol_idx
        bdy_idx = self.v_h.mesh.bdy_idx
    
        # the boundary data are just direct deltas at each node
        bdy_data = np.eye(n_bdy_pts)
        
        # building the rhs for the linear system
        Fb = -self.S_ib@bdy_data
            
        # solve interior dof
        U_vol = spsolve(self.S_ii, Fb)
        
        # allocate the space for the full solution
        sol = np.zeros((n_pts,n_bdy_pts))
        
        # write the corresponding values back to the solution
        sol[bdy_idx,:] = bdy_data
        sol[vol_idx,:] = U_vol

        # computing the flux
        flux = self.S.dot(sol);

        # extracting the boundary data of the flux 
        DtN = flux[bdy_idx, :]

        return DtN, sol

    def adjoint(self, sigma_vec, residual):

        n_bdy_pts = len(self.v_h.mesh.bdy_idx)
        n_pts  = self.v_h.mesh.p.shape[0]
    
        vol_idx = self.v_h.mesh.vol_idx
        bdy_idx = self.v_h.mesh.bdy_idx
        
        # the boundary data are just direct deltas at each node
        bdy_data = residual
        
        # building the rhs for the linear system
        Fb = -self.S_ib@bdy_data
        
        # solve interior dof
        U_vol = spsolve(self.S_ii, Fb)
        
        # allocate the space for the full solution
        sol_adj = np.zeros((n_pts,n_bdy_pts))
        
        # write the corresponding values back to the sol_adjution
        sol_adj[bdy_idx,:] = bdy_data
        sol_adj[vol_idx,:] = U_vol

        return sol_adj 

    def misfit(self, Data, sigma_vec):
        # compute the misfit 

        # compute dtn and sol for given sigma
        dtn, sol = self.dtn_map(sigma_vec)

        # compute the residual
        residual  = -(Data - dtn)

        # comput the adjoint fields
        sol_adj = self.adjoint(sigma_vec, residual)

        Sol_adj_x = self.Dx@sol_adj
        Sol_adj_y = self.Dy@sol_adj

        Sol_x = self.Dx@sol
        Sol_y = self.Dy@sol

        grad = self.M_w@np.sum(Sol_adj_x*Sol_x + Sol_adj_y*Sol_y, axis = 1);

        return 0.5*np.sum(np.square(residual)), grad



def stiffness_matrix_numba(v_h, sigma_vec):
    ''' S = stiffness_matrix(v_h, sigma_vec)
        function to assemble the stiffness matrix 
        for the Poisson equation 
        input: v_h: this contains the information 
               approximation space. For simplicity
               we suppose that the space is piece-wise
               linear polynomials
               sigma_vec: values of sigma at each 
               triangle
    '''
    # define a local handles 
    t = v_h.mesh.t
    p = v_h.mesh.p

    sigma_vec = sigma_vec.astype(np.float64).reshape((-1))

    # we define the arrays for the indicies and the values 
    idx_i = np.zeros((v_h.mesh.n_t, 9), dtype = np.int64)
    idx_j = np.zeros((v_h.mesh.n_t, 9), dtype = np.int64)
    vals = np.zeros((v_h.mesh.n_t, 9), dtype = np.float64)

    # we fill the entried with a numba jitted function
    fill_entries_matrix(idx_i, idx_j, vals, t, p, 
    					sigma_vec, np.int64(t.shape[0]))

    # we add all the indices to make the matrix
    S_coo = spsp.coo_matrix((vals.reshape((-1,)), 
                            (idx_i.reshape((-1,)), 
                             idx_j.reshape((-1,)))), 
    						 shape=(v_h.dim, v_h.dim))

    return spsp.lil_matrix(S_coo) 


@numba.jit(void(int64[:,:], int64, int64[:,:]), nopython=True)
def fill_array(idx, e, matrix):
    for ii in range(3):
        for jj in range(3):
            idx[e, 3*ii+jj] = matrix[ii, jj]


@numba.jit(void(int64[:,:], int64[:,:], float64[:,:], 
				uint16[:,:],  float64[:,:], float64[:], int64),
            nopython=True, parallel=True)
def fill_entries_matrix(idx_i, idx_j, vals, t, p, sigma_vec, size_t):

    # print(idx_i.shape)
    for e in numba.prange(size_t):  # integration over one triangular element at a time
        # row of t = node numbers of the 3 corners of triangle e
        nodes = t[e,:]
        # print(nodes)
  
        # 3 by 3 matrix with rows=[1 xcorner ycorner] 
        Pe = np.concatenate((np.ones((3,1), dtype = np.float64), 
        					 p[nodes,:]), axis = -1)
        # print(Pe)
        # area of triangle e = half of parallelogram area
        Area = np.abs(npla.det(Pe))/2
        # print(Area)
        # columns of C are coeffs in a+bx+cy to give phi=1,0,0 at nodes
        C = npla.inv(Pe); 
        # now compute 3 by 3 Ke and 3 by 1 Fe for element e
        grad = C[1:3,:]
        # element matrix from slopes b,c in grad
        S_local = (sigma_vec[e]*Area)*((grad.T).dot(grad));
        # print(S_local)

        # print(np.ones((3,1), dtype = np.int)*nodes)
        # add S_local  to 9 entries of global K
        # print((np.ones((3,1), dtype = np.int64)*nodes))
        fill_array(idx_i, e, np.ones((3,1), dtype = np.int64)*nodes)
        # print((np.ones((3,1), dtype = np.int64)*nodes).T)
        fill_array(idx_j, e, (np.ones((3,1), dtype = np.int64)*nodes).T)
        # print(idx_j[e,:])
        vals[e,:] = S_local.reshape((9))
        # print(vals[e,:])
