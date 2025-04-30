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


class Domains:
    """Parameters defining the domain of the problem."""
    
    def __init__(self, Lx, Ly, T):
        """Initialize domain parameters."""
        self.Lx = Lx
        self.Ly = Ly
        self.T = T

    def is_boundary(self, x):
        """Check if points are on boundary."""
        is_left = np.isclose(x[:, 0], -self.Lx, atol=1e-10)
        is_right = np.isclose(x[:, 0], self.Lx, atol=1e-10)
        is_bottom = np.isclose(x[:, 1], -self.Ly, atol=1e-10)
        is_top = np.isclose(x[:, 1], self.Ly, atol=1e-10)
        return is_left | is_right | is_bottom | is_top


class Models:
    """Physical model definitions and analytical solution."""
    
    def __init__(self, vx, vy, D, sigma):
        """Initialize model parameters."""
        self.vx = vx
        self.vy = vy
        self.D = D
        self.sigma = sigma
        # Compute Peclet number
        self.peclet = None  # Will be set after mesh creation

    def set_peclet(self, characteristic_length):
        """Compute Peclet number for the problem."""
        v_mag = np.sqrt(self.vx**2 + self.vy**2)
        self.peclet = v_mag * characteristic_length / (2 * self.D)
        print(f"Peclet number: {self.peclet}")

    def analytical_solution(self, xyt):
        """Compute analytical solution at space-time points."""
        # Handle t=0 case separately to avoid division by zero
        t_zero_mask = np.isclose(xyt[:,2], 0.0, atol=1e-14)
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

    def initial_condition_fn(self, xy):
        """Evaluate initial condition."""
        t = np.zeros((xy.shape[0], 1))
        xyt = np.hstack([xy, t])
        return self.analytical_solution(xyt)


class MeshData:
    """Class for storing and processing mesh data."""
    
    def __init__(self, mesh, domains, nt):
        """Initialize mesh data."""
        self.mesh = mesh
        self.domains = domains
        self.nt = nt

        self.time_discr = np.linspace(0, domains.T, nt)

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

        # Compute characteristic length (max edge length)
        self.diameter = 0
        for v1, v2, v3 in self.triangles:
            p1 = self.points[v1]
            p2 = self.points[v2]
            p3 = self.points[v3]
            h = max(np.linalg.norm(p1 - p2), np.linalg.norm(p2 - p3), np.linalg.norm(p3 - p1))
            
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
        """Compute the length of each segment."""
        p = self.points
        lengths = []
        for a, b in self.segments:
            length = np.linalg.norm(p[a] - p[b])
            lengths.append(length)
        return np.array(lengths, dtype=np.float64)

    def _compute_triangle_areas(self):
        """Compute the area of each triangle."""
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
        """Visualize the mesh."""
        plt.figure(figsize=(10, 8))
        plt.triplot(self.points[:, 0], self.points[:, 1], self.triangles)
        plt.axis('equal')
        plt.grid(True)
        plt.title('2D Mesh Visualization')
        plt.show()


class ElementCR:
    def __init__(self):
        self.points = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0]
        ])
        
        self.midpoints = np.array([
             [1/2, 1/2],
             [1/2, 0.0],
             [0.0, 1/2]
        ])
        
        self.segment_enumeration = np.array([
            [1, 2],
            [2, 0],
            [0, 1]
        ])
    
    def get_shape_functions(self, local_coords):
        x, y = local_coords
        return np.array([
            -1 + 2 * (x + y),
            1 - 2 * x,
            1 - 2 * y
        ])
    
    def get_jacobian(self):
        pass
    
    def get_shape_function_derivatives(self):
        return np.array([
            [2.0, 2.0],
            [-2.0, 0.0],
            [0.0, -2.0]
        ])
    
    def get_stiffness_matrix(self):
        return np.array([
            [4.0, -2.0, -2.0],
            [-2.0, 2.0, 0.0],
            [-2.0, 0.0, 2.0]
        ])
    
    def get_mass_matrix(self):
        return np.eye(3) / 6.0
    
# class DirichletBC:
#     """Homogenuous boundary condition"""
#     def __init__(self, boundary_dofs):
#         self.boundary_dofs = boundary_dofs
    
#     def apply(self, S, F):
#         """"Apply Dirichlet BC using the penalty method"""
#         for seg in self.boundary_dofs:
            
            

class BESCRFEM:  # Backward Euler Scheme and Crouzeix-Raviart Finite Element Methods
    """Implementation of Backward Euler scheme with Crouzeix-Raviart FEM."""
    
    def __init__(self, domains, model_params, mesh_data, element, time_scheme_order=1):
        """Initialize solver."""
        self.domains = domains
        self.model_params = model_params
        self.mesh_data = mesh_data
        self.dt = domains.T / (mesh_data.nt - 1)
        self.element = element
        self._compute_reference_element_matrices()
        self.time_scheme_order = time_scheme_order
        
    def _compute_reference_element_matrices(self):
        """Compute reference element matrices analytically."""
        # Stiffness Matrix for gradient dot product
        self.reference_stiffness = self.element.get_stiffness_matrix()
        
        # Mass Matrix for piecewise linear functions
        self.reference_mass = self.element.get_mass_matrix()
        
        # Gradient of Shape Functions
        self.triangle_grad_phis = self.element.get_shape_function_derivatives()

    def compute_stiffness_CR(self, tri_idx):
        """Compute local stiffness matrix for Crouzeix-Raviart element."""
        vertices = self.mesh_data.points[
            self.mesh_data.triangles[tri_idx]
        ]

        # Compute the Jacobian of the transformation
        J = np.zeros((2, 2))
        J[:, 0] = vertices[1, :] - vertices[0, :]
        J[:, 1] = vertices[2, :] - vertices[0, :]
        
        # Determinant of Jacobian (2x area)
        det_J = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
        
        # Inverse of Jacobian
        J_inv = np.array([
            [J[1, 1], -J[0, 1]],
            [-J[1, 0], J[0, 0]]
        ]) / det_J
        
        # Transform reference gradients to physical gradients
        # B_tri = J_inv.T
        # BTB = B_tri @ B_tri.T
        B_tri = J_inv
        BTB = B_tri.T @ B_tri
        
        # Local stiffness matrix
        K_local = self.triangle_grad_phis @ BTB @ self.triangle_grad_phis.T
        return self.model_params.D * self.mesh_data.triangle_areas[tri_idx] * K_local


    def compute_mass_CR(self, tri_idx):
        """Compute local mass matrix for Crouzeix-Raviart element."""
        return self.reference_mass * 2 * self.mesh_data.triangle_areas[tri_idx]
    
    def compute_advection_CR(self, tri_idx):
        #1/ géométrie
        vertices = self.mesh_data.points[
            self.mesh_data.triangles[tri_idx]
        ]
        
        # Compute the Jacobian of the transformation
        J = np.zeros((2, 2))
        J[:, 0] = vertices[1, :] - vertices[0, :]
        J[:, 1] = vertices[2, :] - vertices[0, :]
        
        # Determinant of Jacobian (2x area)
        det_J = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
        
        # Inverse of Jacobian
        B_tri = np.array([
            [J[1, 1], -J[0, 1]],
            [-J[1, 0], J[0, 0]]
        ]) / det_J

        # gradient physique (3×2)
        grad_phi = (B_tri.T @ self.triangle_grad_phis.T).T

        #2/ terme ∫ φ_i (v·∇φ_j) dx = (area/3) * (v·∇φ_j)
        area     = self.mesh_data.triangle_areas[tri_idx]
        v_vec    = np.array([self.model_params.vx, self.model_params.vy])
        phi_int  = np.ones(3) * (area / 6.0)             # φ intégrée
        v_dot_gr = grad_phi @ v_vec                      # (3,)
        A_loc    = np.outer(phi_int, v_dot_gr)           # (3×3)
        return 2 * A_loc
        
        # A_loc = np.zeros((3,3))
        # for i in range(3):
        #     int_phi_i = 1 / 6.0
        #     for j in range(3):
        #         grad_phi_j = self.triangle_grad_phis[j]
        #         v_dot_gr = v_vec.dot(grad_phi_j)
        #         A_loc[j, i] = (2 * area) * v_dot_gr.dot(B_tri) * int_phi_i
        # return A_loc
        
        
    
    def build_global_matrices(self):
        """Build global mass, stiffness and advection matrices"""
        n_seg = self.mesh_data.number_of_segments
        
        #Triplet lists for each matrix
        I_m, J_m, V_m = [], [], []
        I_k, J_k, V_k = [], [], []
        I_a, J_a, V_a = [], [], []
        
        #Loop over through each triangle
        for tri_idx in range(self.mesh_data.number_of_triangles):
            segs = self.mesh_data.triangle_to_segments[tri_idx]
            M_loc = self.compute_mass_CR(tri_idx)
            K_loc = self.compute_stiffness_CR(tri_idx)
            A_loc = self.compute_advection_CR(tri_idx)
            
            # Assemble local -> global
            for a in range(3):
                i = segs[a]
                for b in range(3):
                    j = segs[b]
                    I_m.append(i); J_m.append(j); V_m.append(M_loc[a, b])
                    I_k.append(i); J_k.append(j); V_k.append(K_loc[a, b])
                    I_a.append(i); J_a.append(j); V_a.append(A_loc[a, b])
        
        # Construct CSR sparse matrices
        self.global_mass = csr_matrix((V_m, (I_m, J_m)), shape=(n_seg, n_seg))
        self.global_stiffness = csr_matrix((V_k, (I_k, J_k)), shape=(n_seg, n_seg))
        self.global_advection = csr_matrix((V_a, (I_a, J_a)), shape=(n_seg, n_seg))
        
        #Build system matrix without BC
        if self.time_scheme_order == 1:
            self.base_system = self.global_mass + self.dt * (self.global_stiffness + self.global_advection)
        elif self.time_scheme_order == 2:
            self.base_system = self.global_mass + 0.5 * self.dt * (self.global_stiffness + self.global_advection)
        else:
            raise ValueError(f"Order {self.time_scheme_order} numerical scheme not implemented")

    
    def set_initial_condition(self):
        self.u_prev = self.model_params.initial_condition_fn(self.mesh_data.midpoints)
        
    def set_boundary_fn(self, t):
        # self.mesh_data.midpoints
        n_bc = self.mesh_data.boundary_segments.shape[0]
        
        bc = np.zeros(self.mesh_data.midpoints.shape[0])
        
        t_array = t * np.ones((n_bc, 1))
        
        xyt = np.hstack((self.mesh_data.midpoints[self.mesh_data.boundary_segments], t_array))
        
        # print(self.model_params.analytical_solution(xyt).shape)
        # print(bc.shape)
        bc[self.mesh_data.boundary_segments] = self.model_params.analytical_solution(xyt)
        
        return bc
        
        
    def set_source_term(self):
        if self.time_scheme_order == 1:
            b = self.global_mass.dot(self.u_prev)
        elif self.time_scheme_order == 2:
            b = (self.global_mass - 0.5 * self.dt * (self.global_stiffness + self.global_advection)).dot(self.u_prev)
        else:
            raise ValueError(f"Order {self.time_scheme_order} numerical scheme not implemented")
        
        # Apply boundary condition
        A = self.base_system.copy().tolil()
        
        for seg in self.mesh_data.boundary_segments:
            A.rows[seg] = [seg]
            A.data[seg] = [1.0]
            b[seg] = 0.0
            
        return A.tocsr(), b        
        
    def solve(self):
        # 1. Initial condition and storage
        self.set_initial_condition()
        n_steps = self.mesh_data.nt
        n_segments = self.mesh_data.number_of_segments
        self.solutions = np.zeros((n_steps, n_segments))
        self.solutions[0, :] = self.u_prev
        
        # 2. Assemble global matrices and base system 
        self.build_global_matrices()
        
        # 3. Time-stepping loop
        start = time.time()
        for step in tqdm(range(1, n_steps), desc= "Time-stepping"):
            t = step * self.dt
            
            #get A, b at step t
            A, b = self.set_source_term()
            
            #
            self.u_prev = spsolve(A, b)
            
            #Seting boundary condition by lifting
            self.solutions[step, :] = self.u_prev + self.set_boundary_fn(t)
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
        
        time_values = np.linspace(0, self.domains.T, self.mesh_data.nt)
        
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
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        
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
        
        # # Plot error
        # cntr3 = axs[2].tricontourf(triang, error_values, 20, cmap="coolwarm", 
        #                            norm=plt.Normalize(-np.max(np.abs(error_values)), 
        #                                             np.max(np.abs(error_values))))
        # axs[2].set_title(f"Error at t = {t:.3f}")
        # axs[2].set_xlabel("x")
        # axs[2].set_ylabel("y")
        # fig.colorbar(cntr3, ax=axs[2])
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/solution_t{time_index}_interpolated.png", dpi=300)
        plt.savefig(f"{save_dir}/solution_t{time_index}_interpolated.pdf", dpi=300)
        plt.close()
        print(f"Saved at {save_dir}/solution_t{time_index}_interpolated.png/pdf")
        
        
if __name__ == '__main__':
    domain_size = 20.0
    Lx = Ly = domain_size  # Half-size of the domain
    T = 10.0  # End time
    D = 0.1  # Diffusion coefficient (small value leads to advection-dominated flow)
    vx, vy = 1.0, 0.5  # Velocity field
    sigma = 0.1

    # Create mesh with 30 points per axis (higher resolution)
    mesh_file = create_mesh(128, domain_size=domain_size)
    mesh = meshio.read(mesh_file)

    # Setup parameters
    domains = Domains(Lx=Lx, Ly=Ly, T=T)
    model_params = Models(vx=vx, vy=vy, D=D, sigma=sigma)
    n_steps = 256
    mesh_data = MeshData(mesh, domains, nt=n_steps)  # More time steps for accuracy

    # mesh_data.show()
    print(mesh_data.number_of_segments)


    #Type of finite element methods
    cr_element = ElementCR()

    solver1 = BESCRFEM(domains, model_params, mesh_data, cr_element, 1)

    solutions1 = solver1.solve()

    errors1 = solver1.compute_errors()
    print(f"L2 Error: {errors1['final_l2_error']:0.4f}")

    print(f"Max Error: {errors1['final_linf_error']:0.4f}")

    solver1.plot_interpoleted_solution()
    solver1.plot_solution()
