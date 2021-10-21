import numpy as np
import numpy.linalg as npla
import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve
import scipy.integrate as integrate

import numba
from numba import jit, void, int64, float64, uint16

class Mesh:
    def __init__(self, points, triangles, bdy_idx, vol_idx):
        # self.p    array with the node points (sorted)
        #           type : np.array dim: (n_p, 2)
        # self.n_p  number of node points
        #           type : int
        # self.t    array with indices of points per segment
        #           type : np.array dim: (n_s, 3)
        # self.n_t  number of triangles
        #           type : int
        # self.bc.  array with the indices of boundary points
        #           type : np.array dim: (2)


        self.p = points
        self.t = triangles

        self.n_p = self.p.shape[0]
        self.n_t = self.t.shape[0]

        self.bdy_idx = bdy_idx
        self.vol_idx = vol_idx

        # boundary indices for the triangles in the 
        # boundary 

        # faster search in the for loop
        bdy_idx_set = set(self.bdy_idx)
        
        self.bdy_idx_t = set()
        for e in range(self.t.shape[0]):  # integration over one triangular element at a time
            nodes = self.t[e, :]
            if   (nodes[0] in bdy_idx_set)\
               + (nodes[1] in bdy_idx_set)\
               + (nodes[2] in bdy_idx_set) >= 2:
                self.bdy_idx_t.add(e)


class V_h:
    def __init__(self, mesh):
        # self.mesh Mesh object containg geometric info type: Mesh
        # self.sim  dimension of the space              type: in

        self.mesh = mesh
        self.dim = mesh.n_p


def stiffness_matrix(v_h, sigma_vec):
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

    # we define the arrays for the indicies and the values 
    idx_i = np.zeros((v_h.mesh.n_t, 9), dtype  = np.int)
    idx_j = np.zeros((v_h.mesh.n_t, 9), dtype  = np.int)
    vals = np.zeros((v_h.mesh.n_t, 9), dtype  = np.float64)

    # Assembly the matrix
    for e in range(v_h.mesh.n_t):  # integration over one triangular element at a time
        # row of t = node numbers of the 3 corners of triangle e
        nodes = t[e,:]
  
        # 3 by 3 matrix with rows=[1 xcorner ycorner] 
        Pe = np.concatenate([np.ones((3,1)), p[nodes,:]], axis = -1)
        # area of triangle e = half of parallelogram area
        Area = np.abs(npla.det(Pe))/2
        # columns of C are coeffs in a+bx+cy to give phi=1,0,0 at nodes
        C = npla.inv(Pe); 
        # now compute 3 by 3 Ke and 3 by 1 Fe for element e
        grad = C[1:3,:]
        # element matrix from slopes b,c in grad
        S_local = (sigma_vec[e]*Area)*grad.T.dot(grad);
        
        # add S_local  to 9 entries of global K
        idx_i[e,:] = (np.ones((3,1))*nodes).T.reshape((9,))
        idx_j[e,:] = (np.ones((3,1))*nodes).reshape((9,))
        vals[e,:] = S_local.reshape((9,))

    # we add all the indices to make the matrix
    S_coo = spsp.coo_matrix((vals.reshape((-1,)), 
                            (idx_i.reshape((-1,)), 
                             idx_j.reshape((-1,)))), shape=(v_h.dim, v_h.dim))

    return spsp.lil_matrix(S_coo) 



#####################################################
def mass_matrix(v_h):
    ''' M = mass_matrix(v_h)
        function to assemble the mass matrix 
        for the Poisson equation 
        input: v_h: this contains the information 
               approximation space. For simplicity
               we suppose that the space is piece-wise
               linear polynomials
    '''

    # define a local handles 
    t = v_h.mesh.t
    p = v_h.mesh.p

    idx_i = np.zeros((v_h.mesh.n_t, 9), dtype  = np.int)
    idx_j = np.zeros((v_h.mesh.n_t, 9), dtype  = np.int)
    vals = np.zeros((v_h.mesh.n_t, 9), dtype  = np.float64)

    # local mass matrix (so we don't need to compute it at each iteration)
    MK = 1/12*np.array([ [2., 1., 1.], 
                         [1., 2., 1.],
                         [1., 1., 2.]])


    # Assembly the matrix
    for e in range(v_h.mesh.n_t):  # integration over one triangular element at a time
        # row of t = node numbers of the 3 corners of triangle e
        nodes = t[e,:]
  
        # 3 by 3 matrix with rows=[1 xcorner ycorner] 
        Pe = np.concatenate([np.ones((3,1)), p[nodes,:]], axis = -1)
        # area of triangle e = half of parallelogram area
        Area = np.abs(npla.det(Pe))/2
    
        M_local = Area*MK
        
        # add S_local  to 9 entries of global K
        idx_i[e,:] = (np.ones((3,1))*nodes).T.reshape((9,))
        idx_j[e,:] = (np.ones((3,1))*nodes).reshape((9,))
        vals[e,:] = M_local.reshape((9,))

    # we add all the indices to make the matrix
    M_coo = spsp.coo_matrix((vals.reshape((-1,)), 
                            (idx_i.reshape((-1,)), 
                             idx_j.reshape((-1,)))), shape=(v_h.dim, v_h.dim))

    return spsp.lil_matrix(M_coo) 


def projection_v_w(v_h):
    ''' M = mass_matrix(v_h)
        function to assemble the mass matrix 
        for the Poisson equation 
        input: v_h: this contains the information 
               approximation space. For simplicity
               we suppose that the space is piece-wise
               linear polynomials
    '''

    # define a local handles 
    t = v_h.mesh.t
    p = v_h.mesh.p

    idx_i = np.zeros((v_h.mesh.n_t, 3), dtype  = np.int)
    idx_j = np.zeros((v_h.mesh.n_t, 3), dtype  = np.int)
    vals = np.zeros((v_h.mesh.n_t, 3), dtype  = np.float64)

    # Assembly the matrix
    for e in range(v_h.mesh.n_t):  # integration over one triangular element at a time
        # row of t = node numbers of the 3 corners of triangle e
        nodes = t[e,:]
  
        # 3 by 3 matrix with rows=[1 xcorner ycorner] 
        Pe = np.concatenate([np.ones((3,1)), p[nodes,:]], axis = -1)
        # area of triangle e = half of parallelogram area
        Area = np.abs(npla.det(Pe))/2

        # add S_local  to 9 entries of global K
        idx_i[e,:] = nodes
        idx_j[e,:] = e*np.ones((3,))
        vals[e,:] = np.ones((3,))*Area/3

    # we add all the indices to make the matrix
    M_coo = spsp.coo_matrix((vals.reshape((-1,)), 
                            (idx_i.reshape((-1,)), 
                             idx_j.reshape((-1,)))), 
                            shape=(v_h.dim, v_h.mesh.n_t))

    return spsp.lil_matrix(M_coo) 


def partial_deriv_matrix(v_h):
    ''' Kx, Ky, Surf = mass_matrix(v_h)
        function to assemble the mass matrix 
        for the Poisson equation 
        input: v_h: this contains the information 
               approximation space. For simplicity
               we suppose that the space is piece-wise
               linear polynomials
        output: Kx matrix to compute weak derivatives
                Kx matrix to compute weak derivative
                M_t mass matrix in W (piece-wise constant matrices)
    '''
    # define a local handles 
    t = v_h.mesh.t
    p = v_h.mesh.p

    # number of triangles
    n_t = v_h.mesh.n_t

    idx_i = np.zeros((v_h.mesh.n_t, 3), dtype  = np.int)
    idx_j = np.zeros((v_h.mesh.n_t, 3), dtype  = np.int)
    vals_x = np.zeros((v_h.mesh.n_t, 3), dtype  = np.float64)
    vals_y = np.zeros((v_h.mesh.n_t, 3), dtype  = np.float64)
    vals_s = np.zeros((v_h.mesh.n_t, 1), dtype  = np.float64)

    # Assembly the matrix
    for e in range(n_t):  #
        nodes = t[e,:]
  
        # 3 by 3 matrix with rows=[1 xcorner ycorner] 
        Pe = np.concatenate([np.ones((3,1)), p[nodes,:]], axis = -1)
        # area of triangle e = half of parallelogram area
        Area = np.abs(npla.det(Pe))/2
        # columns of C are coeffs in a+bx+cy to give phi=1,0,0 at nodes
        C = npla.inv(Pe); 
        # now compute 3 by 3 Ke and 3 by 1 Fe for element e
        grad = C[1:3,:]

        Kx_loc = grad[0,:]*Area;
        Ky_loc = grad[1,:]*Area;

        vals_x[e,:] = Kx_loc
        vals_y[e,:] = Ky_loc

        vals_s[e] = Area

        # saving the indices
        idx_i[e,:] = e*np.ones((3,))
        idx_j[e,:] = nodes

    Kx_coo = spsp.coo_matrix((vals_x.reshape((-1,)), 
                             (idx_i.reshape((-1,)), 
                              idx_j.reshape((-1,)))), shape=(n_t, p.shape[0]))

    Ky_coo = spsp.coo_matrix((vals_y.reshape((-1,)), 
                             (idx_i.reshape((-1,)), 
                              idx_j.reshape((-1,)))), shape=(n_t, p.shape[0]))

    surf = spsp.dia_matrix((vals_s.reshape((1,-1)), 
                            np.array([0])), shape=(n_t, n_t))

    return spsp.lil_matrix(Kx_coo), spsp.lil_matrix(Ky_coo), spsp.lil_matrix(surf)  


def dtn_map(v_h, sigma_vec):

    n_bdy_pts = len(v_h.mesh.bdy_idx)
    n_pts  = v_h.mesh.p.shape[0]

    vol_idx = v_h.mesh.vol_idx
    bdy_idx = v_h.mesh.bdy_idx

    # build the stiffness matrix
    S = stiffness_matrix(v_h, sigma_vec)
    
    # reduced Stiffness matrix (only volumetric dof)
    Sb = spsp.csr_matrix(S[vol_idx,:][:,vol_idx])
    
    # the boundary data are just direct deltas at each node
    bdy_data = np.eye(n_bdy_pts)
    
    # building the rhs for the linear system
    Fb = -S[vol_idx,:][:,bdy_idx]*bdy_data
    
    # solve interior dof
    U_vol = spsolve(Sb, Fb)
    
    # allocate the space for the full solution
    sol = np.zeros((n_pts,n_bdy_pts))
    
    # write the corresponding values back to the solution
    sol[bdy_idx,:] = bdy_data
    sol[vol_idx,:] = U_vol

    # computing the flux
    flux = S.dot(sol);

    # extracting the boundary data of the flux 
    DtN = flux[bdy_idx, :]

    return DtN, sol

def adjoint(v_h, sigma_vec, residual):

    n_bdy_pts = len(v_h.mesh.bdy_idx)
    n_pts  = v_h.mesh.p.shape[0]

    vol_idx = v_h.mesh.vol_idx
    bdy_idx = v_h.mesh.bdy_idx

    # build the stiffness matrix
    # given that the operator is self-adjoint
    S = stiffness_matrix(v_h, sigma_vec)
    
    # reduced Stiffness matrix (only volumetric dof)
    Sb = spsp.csr_matrix(S[vol_idx,:][:,vol_idx])
    
    # the boundary data are just direct deltas at each node
    bdy_data = residual
    
    # building the rhs for the linear system
    Fb = -S[vol_idx,:][:,bdy_idx]*bdy_data
    
    # solve interior dof
    U_vol = spsolve(Sb, Fb)
    
    # allocate the space for the full solution
    sol_adj = np.zeros((n_pts,n_bdy_pts))
    
    # write the corresponding values back to the sol_adjution
    sol_adj[bdy_idx,:] = bdy_data
    sol_adj[vol_idx,:] = U_vol

    return sol_adj 


def misfit_sigma(v_h, Data, sigma_vec):
    # compute the misfit 

    # compute dtn and sol for given sigma
    dtn, sol = dtn_map(v_h, sigma_vec)

    # compute the residual
    residual  = -(Data - dtn)

    # comput the adjoint fields
    sol_adj = adjoint(v_h, sigma_vec, residual)

    # compute the derivative matrices (weakly)

    Kx, Ky, M_w = partial_deriv_matrix(v_h)

    # this should be diagonal, thus we can avoid this
    M_w = spsp.csr_matrix(M_w)

    Sol_adj_x = spsolve(M_w,(Kx@sol_adj))
    Sol_adj_y = spsolve(M_w,(Ky@sol_adj))

    Sol_x = spsolve(M_w,(Kx@sol))
    Sol_y = spsolve(M_w,(Ky@sol))

    grad = M_w*np.sum(Sol_adj_x*Sol_x + Sol_adj_y*Sol_y, axis = 1);

    return 0.5*np.sum(np.square(residual)), grad

# if __name__ == "__main__":

#     x = np.linspace(0, 1, 11)

#     mesh = Mesh(x)
#     v_h = V_h(mesh)

#     f_load = lambda x: 2 + 0 * x
#     xi = f_load(x)  # linear function

#     u = Function(xi, v_h)

#     assert np.abs(u(x[5]) - f_load(x[5])) < 1.e-6

#     # check if this is projection
#     ph_f = p_h(v_h, f_load)
#     ph_f2 = p_h(v_h, ph_f)

#     assert np.max(ph_f.xi - ph_f2.xi) < 1.e-6

#     # using analytical solution
#     u = lambda x : np.sin(4*np.pi*x)
#     # building the correct source file
#     f = lambda x : (4*np.pi)**2*np.sin(4*np.pi*x)
#     # conductivity is constant
#     sigma = lambda x : 1 + 0*x  

#     u_sol = solve_poisson_dirichelet(v_h, f, sigma)

#     err = lambda x: np.square(u_sol(x) - u(x))
#     # we use an fearly accurate quadrature 
#     l2_err = np.sqrt(integrate.quad(err, 0.0, 1.)[0])

#     print("L^2 error using %d points is %.6f" % (v_h.dim, l2_err))
#     # this should be quite large

#     # define a finer partition 
#     x = np.linspace(0, 1, 21)
#     # init mesh and fucntion space
#     mesh = Mesh(x)
#     v_h = V_h(mesh)

#     u_sol = solve_poisson_dirichelet(v_h, f, sigma)

#     err = lambda x: np.square(u_sol(x) - u(x))
#     # we use an fearly accurate quadrature
#     l2_err = np.sqrt(integrate.quad(err, 0.0, 1.)[0])

#     # print the error
#     print("L^2 error using %d points is %.6f" % (v_h.dim, l2_err))






