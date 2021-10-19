import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
from pypardiso import spsolve
import scipy.integrate as integrate

import numba
from numba import jit, void, int64, float64, uint16

from .fem import Mesh, V_h, partial_deriv_matrix, mass_matrix
from eit import EIT

class IncompleteEIT(EIT):
    """This is the EIT with incomplete data"""
    def __init__(self, v_h, E):
        # using the inherited contructor
        EIT.__init__(self, v_h)
        # mask with only the available data
        self.E = E

    def adjoint(self, sigma_vec, residual):

        n_bdy_pts = len(self.v_h.mesh.bdy_idx)
        n_pts  = self.v_h.mesh.p.shape[0]
    
        vol_idx = self.v_h.mesh.vol_idx
        bdy_idx = self.v_h.mesh.bdy_idx
        
        # the boundary data are just direct deltas at each node
        bdy_data = self.E*residual
        
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
        # we assume that the data is already being masked
        residual  = -(Data - self.E*dtn)

        # comput the adjoint fields
        sol_adj = self.adjoint(sigma_vec, residual)

        Sol_adj_x = self.Dx@sol_adj
        Sol_adj_y = self.Dy@sol_adj

        Sol_x = self.Dx@sol
        Sol_y = self.Dy@sol

        grad = self.M_w@np.sum(Sol_adj_x*Sol_x + Sol_adj_y*Sol_y, axis = 1);

        return 0.5*np.sum(np.square(residual)), grad
