# Crouzeix-Raviart FEM with Backward Euler for Advection-Diffusion Equation


import meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import gmsh
import sys
import time
from tqdm import tqdm
import os
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


def create_mesh(n_points_per_axis=20, domain_size=2.0, filename="square_mesh.msh"):
    """Create a square mesh using GMSH API."""
    
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("rectangle")
    
    # Parameters of the rectangle
    Lx = Ly = 2*domain_size # Width and height
    x0 = y0 = -domain_size      # Bottom-left corner
    
    # Create rectangle using a built-in function
    gmsh.model.occ.addRectangle(x0, y0, 0, Lx, Ly)
    
    # Synchronize to create the CAD entities
    gmsh.model.occ.synchronize()
    
    # Mesh size (can also define fields for variable mesh)
    mesh_size = (2 * domain_size) / (n_points_per_axis-1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    
    # Generate 2D mesh
    gmsh.model.mesh.generate(2)
    
    # Save mesh file
    gmsh.write(filename)
    
    # Finalize Gmsh
    gmsh.finalize()
    return filename


class DomainParams:
    """Parameters defining the domain of the problem."""
    
    def __init__(self, Lx, Ly, T):
        """Initialize domain parameters."""
        self.Lx = Lx
        self.Ly = Ly
        self.T = T

    def is_boundary(self, x):
        """Check if points are on boundary."""
        is_left = np.isclose(x[:, 0], -self.Lx)
        is_right = np.isclose(x[:, 0], self.Lx)
        is_bottom = np.isclose(x[:, 1], -self.Ly)
        is_top = np.isclose(x[:, 1], self.Ly)
        return is_left | is_right | is_bottom | is_top


class Models:
    """Physical model definitions and analytical solution."""
    
    def __init__(self, vx, vy, D, sigma):
        """Initialize model parameters."""
        self.vx = vx
        self.vy = vy
        self.D = D
        self.sigma = sigma

    def analytical_solution(self, xyt):
        """Compute analytical solution at space-time points."""
        # Handle t=0 case separately to avoid division by zero
        t_zero_mask = xyt[:,2] == 0
        result = np.zeros_like(xyt[:,0])
        
        # For t=0 points
        if np.any(t_zero_mask):
            denom_zero = self.sigma**2
            term_zero = np.exp(
                - (xyt[t_zero_mask,0]**2 + xyt[t_zero_mask,1]**2) / denom_zero
            )
            result[t_zero_mask] = term_zero / (np.pi * denom_zero)
        
        # For t>0 points
        if np.any(~t_zero_mask):
            denom = 4 * self.D * xyt[~t_zero_mask,2] + self.sigma**2
            term = np.exp(
                - ((xyt[~t_zero_mask,0] - xyt[~t_zero_mask,2] * self.vx)**2 + 
                   (xyt[~t_zero_mask,1] - xyt[~t_zero_mask,2] * self.vy)**2) / denom
            )
            result[~t_zero_mask] = term / (np.pi * denom)
            
        return result

    def grad_analytical_solution(self, xyt):
        """Compute gradient of analytical solution."""
        # Handle t=0 case separately
        t_zero_mask = xyt[:,2] == 0
        du_dx = np.zeros_like(xyt[:,0])
        du_dy = np.zeros_like(xyt[:,0])
        
        # For t=0 points
        if np.any(t_zero_mask):
            denom_zero = self.sigma**2
            u_zero = self.analytical_solution(xyt[t_zero_mask])
            du_dx[t_zero_mask] = -2 * xyt[t_zero_mask,0] * u_zero / denom_zero
            du_dy[t_zero_mask] = -2 * xyt[t_zero_mask,1] * u_zero / denom_zero
        
        # For t>0 points
        if np.any(~t_zero_mask):
            denom = 4 * self.D * xyt[~t_zero_mask,2] + self.sigma**2
            u = self.analytical_solution(xyt[~t_zero_mask])
            du_dx[~t_zero_mask] = -2 * (xyt[~t_zero_mask,0] - xyt[~t_zero_mask,2] * self.vx) * u / denom
            du_dy[~t_zero_mask] = -2 * (xyt[~t_zero_mask,1] - xyt[~t_zero_mask,2] * self.vy) * u / denom
            
        return du_dx, du_dy

    def initial_condition_fn(self, xy):
        """Evaluate initial condition."""
        t = np.zeros((xy.shape[0], 1))
        xyt = np.hstack([xy, t])
        return self.analytical_solution(xyt)


class MeshData:
    """Class for storing and processing mesh data."""
    
    def __init__(self, mesh, domain_params, nt):
        """Initialize mesh data."""
        self.mesh = mesh
        self.domain_params = domain_params
        self.nt = nt

        self.time_discr = np.linspace(0, domain_params.T, nt)

        # Points
        self.points = mesh.points[:,:2]
        self.number_of_points = len(self.points)

        # Triangles
        self.triangles = mesh.cells_dict['triangle']
        self.number_of_triangles = len(self.triangles)

        # Segments and mapping
        self.segments, self.triangle_to_segments = self._enumerate_segments()
        self.number_of_segments = len(self.segments)

        # Midpoints
        self.midpoints = (self.points[self.segments[:, 0]] + self.points[self.segments[:, 1]]) / 2.0

        # Compute geometry
        self.segment_lengths = self._compute_segment_lengths()
        self.triangle_areas = self._compute_triangle_areas()

        # Boundary Segments
        segments_flat = self.triangle_to_segments.flatten()
        uniques, counts = np.unique(segments_flat, return_counts=True)
        self.boundary_segments = uniques[counts == 1]

        # Boundary triangles
        self.boundary_triangles = None
        self.boundary_triangle_to_segments = {}

        triangles_with_boundary = []
        
        for idx, triangle_segments in enumerate(self.triangle_to_segments):
            for segment in triangle_segments:
                if segment in self.boundary_segments:
                    triangles_with_boundary.append(idx)
                    self.boundary_triangle_to_segments[idx] = segment
                    break
                    
        self.boundary_triangles = np.array(triangles_with_boundary, dtype=np.int32)

        self.diameter = 0
        for v1, v2, v3 in self.triangles:
            p1 = self.points[v1]
            p2 = self.points[v2]
            p3 = self.points[v3]
            h = max(
                np.linalg.norm(p1 - p2),
                np.linalg.norm(p2 - p3),
                np.linalg.norm(p3 - p1)
            )
            if self.diameter < h:
                self.diameter = h

    
    def _enumerate_segments(self):
        """Enumerate all segments in the mesh."""
        segment_map = {}
        triangle_to_segments = []
        segment_id = 0

        for tri in self.triangles:
            tri_segments = []
            edges = [(tri[1], tri[2]), (tri[2], tri[0]), (tri[0], tri[1])]

            for a, b in edges:
                edge = tuple(sorted((a, b)))
                if edge not in segment_map:
                    segment_map[edge] = segment_id
                    segment_id += 1
                tri_segments.append(segment_map[edge])

            triangle_to_segments.append(tri_segments)

        segments = np.array(list(segment_map.keys()), dtype=np.int32)
        triangle_to_segments = np.array(triangle_to_segments, dtype=np.int32)

        return segments, triangle_to_segments
    

    def _compute_segment_lengths(self):
        """Compute the length of each segment.        """
        p = self.points
        lengths = []
        for a, b in self.segments:
            length = np.linalg.norm(p[a] - p[b])
            lengths.append(length)
        return np.array(lengths, dtype=np.float64)

    def _compute_triangle_areas(self):
        """Compute the area of each triangle. """
        p = self.points
        areas = []
        for i, j, k in self.triangles:
            # Use determinant-based formula for triangle area
            x1, y1 = p[i]
            x2, y2 = p[j]
            x3, y3 = p[k]
            area = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
            areas.append(area)
        return np.array(areas, dtype=np.float64)

    def show(self):
        # Créer la figure
        plt.figure(figsize=(10, 8))
        plt.triplot(self.points[:, 0], self.points[:, 1], self.triangles)
        plt.axis('equal')
        plt.grid(True)
        plt.title('2D Mesh Visualisation')
        plt.show()


class BESCRFEM:  # Backward Euler Scheme and Crouzeix-Raviart Finite Element Methods
    """Implementation of Backward Euler scheme with Crouzeix-Raviart FEM."""
    
    def __init__(self, domain_params, model_params, mesh_data, use_quadrature=False):
        """Initialize solver."""
        self.domain_params = domain_params
        self.model_params = model_params
        self.mesh_data = mesh_data
        self.dt = domain_params.T / (mesh_data.nt - 1)

        # Reference Matrices
        self.triangle_grad_phis = np.array([
            [2.0, 2.0],
            [-2.0, 0.0],
            [0.0, -2.0]
        ], dtype=np.float64)

        # ==== Quadrature d'ordre 5 sur l'élément de référence ====
        if use_quadrature:
            self._compute_reference_element_matrices_order5()
        else:
            self._compute_reference_element_matrices()

    def _compute_reference_element_matrices(self):
        # ==== Stiffness Matrix ====
            self.reference_stiffness = np.array([
                [4.0, -2.0, -2.0],
                [-2.0, 2.0, 0.0],
                [-2.0, 0.0, 2.0]
            ], dtype=np.float64)
        
            # ==== Mass Matrix =====
            self.reference_mass = np.array([
                [1, 0, 0],
                [1, 1, 0],
                [0, 0, 1]
            ]) / 6.0

    def _compute_reference_element_matrices_order5(self):
        """Calcule mass, stiffness et advection locales via une quadrature de degré 5."""
        import numpy as np, math
        # Règle de Hammer-Stroud d'ordre 5 sur triangle de référence (0,0),(1,0),(0,1)
        sqrt15 = math.sqrt(15.0)
        # barycentriques transformées en (x,y)
        a1 = (6 + sqrt15) / 21
        b1 = (9 - 2*sqrt15) / 21
        a2 = (6 - sqrt15) / 21
        b2 = (9 + 2*sqrt15) / 21
        pts = np.array([
            [1/3,    1/3   ],
            [a1,     a1    ],
            [a1,     b1    ],
            [b1,     a1    ],
            [a2,     a2    ],
            [a2,     b2    ],
            [b2,     a2    ]
        ])
        wts = np.array([
            9/80,
            (155+sqrt15)/2400,
            (155+sqrt15)/2400,
            (155+sqrt15)/2400,
            (155-sqrt15)/2400,
            (155-sqrt15)/2400,
            (155-sqrt15)/2400
        ])
        # Fonctions de forme CR sur l'élément de référence
        phis = [
            lambda x,y: -1 + 2*x + 2*y,
            lambda x,y:  1 - 2*x,
            lambda x,y:  1 - 2*y
        ]
        grads = np.array([[2,2],[-2,0],[0,-2]], dtype=float)

        # Initialisation
        M_ref = np.zeros((3,3))
        K_ref = np.zeros((3,3))
        A_ref = np.zeros((3,3))
        vx, vy = self.model_params.vx, self.model_params.vy

        # Boucle de quadrature
        for (xi, yi), w in zip(pts, wts):
            phi_vals = np.array([phi(xi, yi) for phi in phis])
            for i in range(3):
                for j in range(3):
                    # masse
                    M_ref[i,j] += w * phi_vals[i] * phi_vals[j]
                    # rigidité (stiffness)
                    K_ref[i,j] += w * np.dot(grads[i], grads[j])
                    # advection
                    v_dot_grad = vx*grads[i,0] + vy*grads[i,1]
                    A_ref[i,j] += w * v_dot_grad * phi_vals[j]

        # Stockage
        self.reference_mass      = M_ref
        self.reference_stiffness = K_ref
        self.reference_advection = A_ref

    def compute_stiffness_CR(self, tri_idx):
        """Compute local stiffness matrix for Crouzeix-Raviart element."""
        vertices = self.mesh_data.points[
            self.mesh_data.triangles[tri_idx]
        ]

        # compute jacobian of the transformation
        A_tri = (vertices[1:, :] - vertices[0, :]).T

        # compute the jacobian's inverse
        B_tri = np.linalg.solve(A_tri, np.eye(2))
        
        # compute B_T^T * B_T for the transformed gradients
        BTB = B_tri.T @ B_tri

        # compute local stiffness matrix for current element
        K_local = self.triangle_grad_phis @ BTB @ self.triangle_grad_phis.T        
        return self.model_params.D * self.mesh_data.triangle_areas[tri_idx] * K_local

    def compute_mass_CR(self, tri_idx):
        """Compute local mass matrix for Crouzeix-Raviart element."""
        return self.reference_mass * 2 * self.mesh_data.triangle_areas[tri_idx]

    def compute_advection_CR(self, tri_idx):
        # 1) géométrie
        verts = self.mesh_data.points[self.mesh_data.triangles[tri_idx]]
        A_tri = (verts[1:] - verts[0]).T
        B_tri = np.linalg.solve(A_tri, np.eye(2))        # J^{-1}
        # gradient physique (3×2)
        grad_phi = (B_tri.T @ self.triangle_grad_phis.T).T
    
        # 2) terme ∫ φ_i (v·∇φ_j) dx = (area/3) * (v·∇φ_j)
        area     = self.mesh_data.triangle_areas[tri_idx]
        phi_int  = np.ones(3) * (area / 3.0)             # φ intégrée
        v_vec    = np.array([self.model_params.vx, self.model_params.vy])
        v_dot_gr = grad_phi @ v_vec                      # (3,)
        A_loc    = np.outer(phi_int, v_dot_gr)           # (3×3)
        return 2 * A_loc

        
    def build_global_matrices(self):
        """Build global mass, stiffness and advection matrices via triplet assembly."""
        n_seg = self.mesh_data.number_of_segments
        # listes de triplets pour chacune des matrices
        I_m, J_m, V_m = [], [], []
        I_k, J_k, V_k = [], [], []
        I_a, J_a, V_a = [], [], []

        # boucle sur chaque triangle
        for tri_idx in range(self.mesh_data.number_of_triangles):
            segs = self.mesh_data.triangle_to_segments[tri_idx].tolist()  # [i0,i1,i2]
            M_loc = self.compute_mass_CR(tri_idx)
            K_loc = self.compute_stiffness_CR(tri_idx)
            A_loc = self.compute_advection_CR(tri_idx)

            # assemble local -> global
            for a in range(3):
                i = segs[a]
                for b in range(3):
                    j = segs[b]
                    I_m.append(i); J_m.append(j); V_m.append(M_loc[a, b])
                    I_k.append(i); J_k.append(j); V_k.append(K_loc[a, b])
                    I_a.append(i); J_a.append(j); V_a.append(A_loc[a, b])

        # construction des sparse CSR
        self.global_mass      = csr_matrix((V_m, (I_m, J_m)), shape=(n_seg, n_seg))
        self.global_stiffness = csr_matrix((V_k, (I_k, J_k)), shape=(n_seg, n_seg))
        self.global_advection = csr_matrix((V_a, (I_a, J_a)), shape=(n_seg, n_seg))

        # build system matrix once (sans BC)
        self.base_system = self.global_mass.copy() / self.dt \
                         + self.global_advection       \
                         + self.global_stiffness

    
    def set_initial_condition(self):
        """
        Set initial condition.
        """
        self.u_prev = self.model_params.initial_condition_fn(self.mesh_data.midpoints)

    def set_source_term(self, tt):
        # Quadrature d'ordre 5 sur [0, 1] pour segment
        quad_pts_1d = np.array([
            0.5 - np.sqrt(5 + 2*np.sqrt(10/7))/6,
            0.5 - np.sqrt(5 - 2*np.sqrt(10/7))/6,
            0.5,
            0.5 + np.sqrt(5 - 2*np.sqrt(10/7))/6,
            0.5 + np.sqrt(5 + 2*np.sqrt(10/7))/6
        ])

        quad_wts_1d = np.array([
            0.2369268851,
            0.4786286705,
            0.5688888889,
            0.4786286705,
            0.2369268851
        ])

        # 1) Second membre non modifié : M/dt * u^n
        b = self.global_mass.dot(self.u_prev) / self.dt

        # 2) Copier base_system et passer en LIL pour poser les BC
        A = self.base_system.copy().tolil()

        # 3) Imposer Dirichlet sur chaque segment frontière

        for seg in self.mesh_data.boundary_segments:
            # Récupération des extrémités du segment
            p0, p1 = self.mesh_data.points[
                self.mesh_data.segments[seg],:
            ]  # (2,) chacun
            

            # Coordonnées physiques des points de quadrature sur le segment
            seg_pts = np.outer(1 - quad_pts_1d, p0) + np.outer(quad_pts_1d, p1)  # (5, 2)
            xyt = np.hstack((seg_pts, np.full((5,1), tt)))  # (5, 3)

            # Évaluer la condition de Dirichlet au temps t
            vals = np.array([self.model_params.analytical_solution(pt.reshape(1,3)).item() for pt in xyt])
            
            # Intégrale approchée sur le segment (longueur * somme pondérée)
            seg_len = np.linalg.norm(p1 - p0)
            bc_val = np.dot(vals, quad_wts_1d) * seg_len  # approximation intégrale
            
            # Normaliser par la longueur du segment pour obtenir une moyenne pondérée
            bc_val /= np.sum(quad_wts_1d) * seg_len

            # Imposer la condition comme avant
            A.rows[seg] = [seg]
            A.data[seg] = [1.0]
            b[seg] = bc_val
        
        return A.tocsr(), b

    def solve(self):
        """  
        Schéma de Backward Euler + FEM CR :
        - assemble une fois base_system via build_global_matrices()
        - à chaque pas : set_source_term(t) → (A,b), puis spsolve
        """
        # 1) Initial condition and storage
        self.set_initial_condition()  
        n_steps    = self.mesh_data.nt
        n_segments = self.mesh_data.number_of_segments
        self.solutions = np.zeros((n_steps, n_segments))
        self.solutions[0, :] = self.u_prev

        # 2) Assemble global matrices and base system (sans BC)
        self.build_global_matrices()  
        # build_global_matrices doit avoir défini self.base_system

        # 3) Time‐stepping loop
        start = time.time()
        for step in tqdm(range(1, n_steps), desc="Time‐stepping"):
            t = step * self.dt

            # récupérer A, b au pas t
            A, b = self.set_source_term(t)

            # résoudre (M/dt + A + K) u^{n+1} = b
            self.u_prev = spsolve(A, b)
            self.solutions[step, :] = self.u_prev

        # 4) Reporting
        self.solve_time = time.time() - start
        print(f"Solve completed in {self.solve_time:.2f}s")

        return self.solutions

    
    def compute_errors(self):
        """Compute errors between numerical and analytical solutions."""
        n_steps = self.mesh_data.nt
        n_segments = self.mesh_data.number_of_segments
        
        # Initialize error metrics
        l2_errors = np.zeros(n_steps)
        linf_errors = np.zeros(n_steps)
        
        # For each time step, compute errors
        for i in range(n_steps):
            t = i * self.dt
            
            # Get numerical solution at this time step
            numerical_sol = self.solutions[i, :]
            
            # Compute analytical solution at midpoints at time t
            midpoints = self.mesh_data.midpoints
            t_array = np.full((n_segments, 1), t)
            xyt = np.hstack([midpoints, t_array])
            analytical_sol = self.model_params.analytical_solution(xyt)
            
            # Compute error
            error = numerical_sol - analytical_sol
            
            # Compute segment volumes (areas) for L2 norm
            segment_volumes = self.mesh_data.segment_lengths / 2  # Half-length for 1D segments
            
            # Compute L2 error (weighted by segment volumes)
            l2_errors[i] = np.sqrt(np.sum(error**2 * segment_volumes) / np.sum(segment_volumes))
            
            # Compute L∞ error
            linf_errors[i] = np.max(np.abs(error))
        
        return {
            'l2_errors': l2_errors,
            'linf_errors': linf_errors,
            'final_l2_error': l2_errors[-1],
            'final_linf_error': linf_errors[-1]
        }

    def plot_solution(self, time_index=None, save_dir="results"):
        """Plot solution at specified time index."""
        if time_index is None:
            time_index = self.mesh_data.nt - 1
        
        t = time_index * self.dt
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # # Get points and triangles
        # points = self.mesh_data.points
        # triangles = self.mesh_data.triangles
        midpoints = self.mesh_data.midpoints
        triangle_to_segments = self.mesh_data.triangle_to_segments
        
        # Get numerical solution at the specified time index
        numerical_midpoint_values = self.solutions[time_index]
        
        # Compute analytical solution at vertices
        t_array = np.full((len(midpoints), 1), t)
        xyt = np.hstack([midpoints, t_array])
        analytical_midpoint_values = self.model_params.analytical_solution(xyt)
        
        # Error at vertices
        error_values = numerical_midpoint_values - analytical_midpoint_values
        
        # Create subplot
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot numerical solution
        triang = mtri.Triangulation(midpoints[:, 0], midpoints[:, 1], triangle_to_segments)
        cntr1 = axs[0].tricontourf(triang, numerical_midpoint_values, 20, cmap="viridis")
        axs[0].set_title(f"Numerical Solution at t = {t:.3f}")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        fig.colorbar(cntr1, ax=axs[0])
        
        # Plot analytical solution
        cntr2 = axs[1].tricontourf(triang, analytical_midpoint_values, 20, cmap="viridis")
        axs[1].set_title(f"Analytical Solution at t = {t:.3f}")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        fig.colorbar(cntr2, ax=axs[1])
        
        # Plot error
        cntr3 = axs[2].tricontourf(triang, error_values, 20, cmap="coolwarm", 
                                   norm=plt.Normalize(-np.max(np.abs(error_values)), 
                                                    np.max(np.abs(error_values))))
        axs[2].set_title(f"Error at t = {t:.3f}")
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("y")
        fig.colorbar(cntr3, ax=axs[2])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/solution_t{time_index}.png", dpi=300)
        plt.close()

    def plot_error_evolution(self, errors, save_dir="results"):
        """Plot error evolution over time."""
        os.makedirs(save_dir, exist_ok=True)
        
        time_values = np.linspace(0, self.domain_params.T, self.mesh_data.nt)
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(time_values, errors['l2_errors'], 'b-', label="L2 Error")
        plt.semilogy(time_values, errors['linf_errors'], 'r-', label="L∞ Error")
        plt.grid(True)
        plt.xlabel("Time")
        plt.ylabel("Error (log scale)")
        plt.title("Error Evolution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/error_evolution.png", dpi=300)
        plt.close()

    def visualize_solution(self, step_idx=None):
        """
        Visualiser la solution pour un pas de temps donné
        """
        if step_idx is None:
            step_idx = self.mesh_data.nt - 1  # Visualiser la dernière étape par défaut

        # Convertir les points du maillage en arrays numpy
        points = self.mesh_data.points
        triangles = self.mesh_data.triangles
        
        # Pour visualiser la solution sur les triangles, nous interpolons la solution
        # des milieux des segments vers les sommets des triangles
        vertex_values = np.zeros(len(points))
        vertex_counts = np.zeros(len(points))

        # Pour chaque segment, ajouter sa valeur aux sommets correspondants
        for i, (a, b) in enumerate(self.mesh_data.segments):
            vertex_values[a] += self.solutions[step_idx, i]
            vertex_values[b] += self.solutions[step_idx, i]
            vertex_counts[a] += 1
            vertex_counts[b] += 1

        # Moyenner les valeurs
        vertex_values /= np.maximum(vertex_counts, 1)

        # Créer la figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Créer une triangulation pour la visualisation
        triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

        # Tracer la solution
        contour = ax.tricontourf(triang, vertex_values, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='Concentration')

        # Ajouter un titre avec le temps
        t = step_idx * self.dt
        ax.set_title(f'Solution at t = {t:.2f} s')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        plt.tight_layout()
        return fig

    def plot_interpoleted_solution(self, time_index=None, save_dir="results"):
        """
        Plot solution at specified time index.
        
        Parameters:
        -----------
        time_index: int or None
            Time index to plot. If None, plot final time.
        save_dir: str
            Directory to save plots
        """
        if time_index is None:
            time_index = self.mesh_data.nt - 1
        
        t = time_index * self.dt
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Get points and triangles
        points = self.mesh_data.points
        triangles = self.mesh_data.triangles
        
        # Get numerical solution at the specified time index
        numerical_sol = self.solutions[time_index]
        
        # Map segment values to vertices for plotting
        vertex_values = np.zeros(len(points))
        count = np.zeros(len(points))
        
        for i, (a, b) in enumerate(self.mesh_data.segments):
            vertex_values[a] += numerical_sol[i]
            vertex_values[b] += numerical_sol[i]
            count[a] += 1
            count[b] += 1
        
        # Average values at vertices
        vertex_values /= np.maximum(count, 1)  # Avoid division by zero
        
        # Compute analytical solution at vertices
        # t_tensor = torch.full((len(points), 1), t, device=device)
        # xyt = torch.cat([self.mesh_data.points, t_tensor], dim=1)
        # analytical_vertex_values = self.model_params.analytical_solution(xyt).cpu().numpy()
        # Compute analytical solution at vertices
        t_array = np.full((len(points), 1), t)
        xyt = np.hstack([points, t_array])
        analytical_vertex_values = self.model_params.analytical_solution(xyt)

        
        # Error at vertices
        error_values = vertex_values - analytical_vertex_values
        
        # Create subplot
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot numerical solution
        triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
        cntr1 = axs[0].tricontourf(triang, vertex_values, 20, cmap="viridis")
        axs[0].set_title(f"Numerical Solution at t = {t:.3f}")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        fig.colorbar(cntr1, ax=axs[0])
        
        # Plot analytical solution
        cntr2 = axs[1].tricontourf(triang, analytical_vertex_values, 20, cmap="viridis")
        axs[1].set_title(f"Analytical Solution at t = {t:.3f}")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        fig.colorbar(cntr2, ax=axs[1])
        
        # Plot error
        cntr3 = axs[2].tricontourf(triang, error_values, 20, cmap="coolwarm", 
                                   norm=plt.Normalize(-np.max(np.abs(error_values)), 
                                                    np.max(np.abs(error_values))))
        axs[2].set_title(f"Error at t = {t:.3f}")
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("y")
        fig.colorbar(cntr3, ax=axs[2])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/solution_t{time_index}.png", dpi=300)
        plt.close()


def convergence_analysis(domain_params, model_params, domain_size, mesh_sizes, nt_values, save_dir="results"):
    """Perform convergence analysis."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize results storage
    results = []
    
    # Loop over mesh sizes and time steps
    for mesh_size in mesh_sizes:
        for nt in nt_values:
            print(f"Running with mesh size {mesh_size} and {nt} time steps")
            
            # Create mesh
            mesh_file = create_mesh(n_points_per_axis=mesh_size, domain_size=domain_size)
            mesh = meshio.read(mesh_file)
            
            # Setup mesh data
            mesh_data = MeshData(mesh, domain_params, nt)
            
            # Initialize solver
            solver = BESCRFEM(domain_params, model_params, mesh_data)
            
            # Solve
            solver.solve()
            
            # Compute errors
            errors = solver.compute_errors()
            
            # Save results
            result = {
                'mesh_size': mesh_size,
                'nt': nt,
                'h': (2 * domain_size) / (mesh_size - 1),  # Approximate element size
                'dt': domain_params.T / (nt - 1),
                'l2_error': errors['final_l2_error'],
                'linf_error': errors['final_linf_error'],
                'compute_time': solver.solve_time
            }
            results.append(result)
            
            # Plot final solution
            solver.plot_solution(save_dir=f"{save_dir}/mesh{mesh_size}_nt{nt}")
            
            # Plot error evolution
            solver.plot_error_evolution(errors, save_dir="results")


class BESCRFEMUpwind(BESCRFEM):
    """
    Sous-classe de BESCRFEM introduisant plusieurs schémas d'upwind pour la convection.

    Paramètres supplémentaires:
    - upwind_type: 'centered' (pas d'upwind), 'edge_upwind' (flux upstream), 'SUPG' (Petrov-Galerkin)
    """
    def __init__(self, domain_params, model_params, mesh_data, upwind_type='edge_upwind'):
        super().__init__(domain_params, model_params, mesh_data)
        self.upwind_type = upwind_type.lower()

    def compute_advection_CR(self, tri_idx):
        if self.upwind_type == 'centered':
            return super().compute_advection_CR(tri_idx)
        elif self.upwind_type == 'edge_upwind':
            return self._compute_edge_upwind(tri_idx)
        elif self.upwind_type == 'supg':
            return self._compute_supg(tri_idx)
        else:
            raise ValueError(f"Unknown upwind_type: {self.upwind_type}")

    def _compute_edge_upwind(self, tri_idx):
        """
        Upwind simple par flux sur chaque arête (first-order upwind).
        Contribution locale diagonale: sum_e max(v·n_e,0)*|e|.
        """
        pts = self.mesh_data.points[self.mesh_data.triangles[tri_idx]]  # (3,2)
        v = np.array([self.model_params.vx, self.model_params.vy])
        # liste des arêtes locales (indice i correspond à arête opposée au noeud i)
        edges = [(pts[1], pts[2]), (pts[2], pts[0]), (pts[0], pts[1])]
        A_loc = np.zeros((3,3))
        for i, (p, q) in enumerate(edges):
            # vecteur arête
            e = q - p
            # normal non orienté
            n = np.array([e[1], -e[0]])
            # normalisation par longueur
            length = np.linalg.norm(e)
            if length == 0:
                continue
            n_unit = n / length
            # flux positif (inflow si v·n>0)
            flux = max(np.dot(v, n_unit), 0.0)
            # upwind matrix diag
            A_loc[i,i] += flux * length
        return A_loc

    def _compute_supg(self, tri_idx):
        """
        Stabilisation SUPG: on ajoute tau * (v·∇φ_i)(v·∇φ_j) * |K|.
        tau = h / (2|v|), avec h = 2*area / perimeter
        """
        # jacobien et aire
        verts = self.mesh_data.points[self.mesh_data.triangles[tri_idx]]
        A_tri = (verts[1:] - verts[0]).T
        area = self.mesh_data.triangle_areas[tri_idx]
        # calcul de h via périmètre
        perim = 0.0
        for a,b in [(1,2),(2,0), (0,1)]:
            perim += np.linalg.norm(verts[b] - verts[a])
        h = 2*area / perim if perim>0 else 0.0
        v = np.array([self.model_params.vx, self.model_params.vy])
        vnorm = np.linalg.norm(v)
        tau = h / (2*vnorm) if vnorm>0 else 0.0
        # gradients physiques
        B_tri = np.linalg.solve(A_tri, np.eye(2))
        grad_phi = self.triangle_grad_phis @ B_tri
        # vecteur v·grad pour chaque base
        v_dot_grad = grad_phi.dot(v)
        # terme SUPG local
        A_supg = tau * area * np.outer(v_dot_grad, v_dot_grad)
        # ajouter terme centré
        A_centered = super().compute_advection_CR(tri_idx)
        return A_centered + A_supg

    def build_global_matrices(self):
        """Assemble global_mass, global_stiffness, global_advection et base_system"""
        n_seg = self.mesh_data.number_of_segments
        I_m, J_m, V_m = [], [], []
        I_k, J_k, V_k = [], [], []
        I_a, J_a, V_a = [], [], []
        for t in range(self.mesh_data.number_of_triangles):
            segs = self.mesh_data.triangle_to_segments[t].tolist()
            Mloc = self.compute_mass_CR(t)
            Kloc = self.compute_stiffness_CR(t)
            Aloc = self.compute_advection_CR(t)
            for i in range(3):
                for j in range(3):
                    I_m.append(segs[i]); J_m.append(segs[j]); V_m.append(Mloc[i,j])
                    I_k.append(segs[i]); J_k.append(segs[j]); V_k.append(Kloc[i,j])
                    I_a.append(segs[i]); J_a.append(segs[j]); V_a.append(Aloc[i,j])
        self.global_mass      = csr_matrix((V_m, (I_m, J_m)), shape=(n_seg, n_seg))
        self.global_stiffness = csr_matrix((V_k, (I_k, J_k)), shape=(n_seg, n_seg))
        self.global_advection = csr_matrix((V_a, (I_a, J_a)), shape=(n_seg, n_seg))
        self.base_system = self.global_mass.copy()/self.dt + self.global_advection + self.global_stiffness

if __name__ == "__main__":
    domain_size = 20.0
    Lx = Ly = domain_size  # Half-size of the domain
    T = 10.0  # End time
    D = 0.1  # Diffusion coefficient (small value leads to advection-dominated flow)
    vx, vy = 1.0, 0.5  # Velocity field
    sigma = 0.1
        
    # Create mesh with 30 points per axis (higher resolution)
    mesh_file = create_mesh(64, domain_size=domain_size)
    mesh = meshio.read(mesh_file)
    
    # Setup parameters
    domain_params = DomainParams(Lx=Lx, Ly=Ly, T=T)
    model_params = Models(vx=vx, vy=vy, D=D, sigma=sigma)
    n_steps = 128
    mesh_data = MeshData(mesh, domain_params, nt=n_steps)  # More time steps for accuracy
    
    # mesh_data.show()
    print(mesh_data.number_of_segments)
    
    bescrfem = BESCRFEM(domain_params, model_params, mesh_data, use_quadrature=True)
    bescrfem.solve()
    
    errors = bescrfem.compute_errors()
    print(f"L2 Error: {errors['final_l2_error']:0.4f}")
    print(f"Max Error: {errors['final_linf_error']:0.4f}")
