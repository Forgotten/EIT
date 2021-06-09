class EllipticSolver:
    def __init__(self, v_h):
        self.v_h = v_h


    def update_matrices(self, sigma_vec):

        vol_idx = v_h.mesh.vol_idx
        bdy_idx = v_h.mesh.bdy_idx

        self.S  = stiffness_matrix(self.v_h, sigma_vec)

    def build_matrices 

        self.Mass = mass_matrix(self.v_h)
        # self.Kx
        # self.Ky 


    def dtn_map(self):
        # do this here

