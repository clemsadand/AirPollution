import torch
import numpy as np
import crbe
from crbe import AdDifProblem
import pinn
import meshio
from tqdm import tqdm
import time
import psutil
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.spatial.distance import cdist
import seaborn as sns

torch.manual_seed(1234)
np.random.seed(1234)


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def backend(x):
    if isinstance(x, np.ndarray):
        return np
    elif isinstance(x, torch.Tensor):
        return torch
    else:
        raise TypeError("Unsupported type")

class Problem(AdDifProblem):
    def __init__(self, v=[1.0, 0.0], D=0.1):
        super().__init__(v, D)

    def initial_condition_fn(self, xy):
        xp = backend(xy)
        cond_x = (xy[:, 0] >= 8.0) & (xy[:, 0] <= 12.0)
        cond_y = (xy[:, 1] >= 8.0) & (xy[:, 1] <= 12.0)
        return xp.where(cond_x & cond_y, xp.ones_like(xy[:, 0]), xp.zeros_like(xy[:, 0]))
    
    def boundary_fn(self, xyt):
        xp = backend(xyt)
        return xp.zeros_like(xyt[:, 0])
    
    def source_term(self, xyt):
        xp = backend(xyt)
        return xp.zeros_like(xyt[:, 0])

class ComprehensiveAnalysis:
    def __init__(self, problem, domain, mesh_data, solver_crbe, model_pinn):
        self.problem = problem
        self.domain = domain
        self.mesh_data = mesh_data
        self.solver_crbe = solver_crbe
        self.model_pinn = model_pinn
        self.results = {}
        
    def compute_mass_conservation(self):
        """Analyze mass conservation over time for both methods using triangle-based integration"""
        print("Computing mass conservation analysis...")
        
        times = self.mesh_data.time_discr
        crbe_masses = []
        pinn_masses = []
        
        for i, t in enumerate(times):
            crbe_mass = 0.0
            pinn_mass = 0.0
            
            # Integrate over triangles
            for tri_idx in range(self.mesh_data.number_of_triangles):
                segs = self.mesh_data.triangle_to_segments[tri_idx]
                area = self.mesh_data.triangle_areas[tri_idx]
                
                # CRBE solution on triangle segments
                crbe_solution_tri = self.solver_crbe.solutions[i, segs]
                crbe_mass += area * np.sum(crbe_solution_tri) / 3
                
                # PINN solution on triangle segments
                midpoints_tri = self.mesh_data.midpoints[segs, :]
                midpoints_tensor = torch.tensor(midpoints_tri, dtype=torch.float32, device=device)
                t_tensor = torch.full((len(segs), 1), t, dtype=torch.float32, device=device)
                xyt = torch.cat([midpoints_tensor, t_tensor], dim=1)
                
                with torch.no_grad():
                    pinn_solution_tri = self.model_pinn(xyt).squeeze().detach().cpu().numpy()
                
                pinn_mass += area * np.sum(pinn_solution_tri) / 3
            
            crbe_masses.append(crbe_mass)
            pinn_masses.append(pinn_mass)
        
        # Store results
        self.results['mass_conservation'] = {
            'times': times,
            'crbe_masses': np.array(crbe_masses),
            'pinn_masses': np.array(pinn_masses),
            'initial_mass': crbe_masses[0]
        }
        
        return self.results['mass_conservation']
    
    def compute_center_of_mass_tracking(self):
        """Track the center of mass movement for both methods using triangle-based integration"""
        print("Computing center of mass tracking...")
        
        times = self.mesh_data.time_discr
        crbe_com_x, crbe_com_y = [], []
        pinn_com_x, pinn_com_y = [], []
        
        for i, t in enumerate(times):
            crbe_mass_total = 0.0
            crbe_moment_x = 0.0
            crbe_moment_y = 0.0
            
            pinn_mass_total = 0.0
            pinn_moment_x = 0.0
            pinn_moment_y = 0.0
            
            # Integrate over triangles
            for tri_idx in range(self.mesh_data.number_of_triangles):
                segs = self.mesh_data.triangle_to_segments[tri_idx]
                area = self.mesh_data.triangle_areas[tri_idx]
                
                # CRBE center of mass calculation
                crbe_solution_tri = self.solver_crbe.solutions[i, segs]
                midpoints_tri = self.mesh_data.midpoints[segs, :]
                
                local_mass_crbe = area * np.sum(crbe_solution_tri) / 3
                local_moment_x_crbe = area * np.sum(crbe_solution_tri * midpoints_tri[:, 0]) / 3
                local_moment_y_crbe = area * np.sum(crbe_solution_tri * midpoints_tri[:, 1]) / 3
                
                crbe_mass_total += local_mass_crbe
                crbe_moment_x += local_moment_x_crbe
                crbe_moment_y += local_moment_y_crbe
                
                # PINN center of mass calculation
                midpoints_tensor = torch.tensor(midpoints_tri, dtype=torch.float32, device=device)
                t_tensor = torch.full((len(segs), 1), t, dtype=torch.float32, device=device)
                xyt = torch.cat([midpoints_tensor, t_tensor], dim=1)
                
                with torch.no_grad():
                    pinn_solution_tri = self.model_pinn(xyt).squeeze().detach().cpu().numpy()
                
                local_mass_pinn = area * np.sum(pinn_solution_tri) / 3
                local_moment_x_pinn = area * np.sum(pinn_solution_tri * midpoints_tri[:, 0]) / 3
                local_moment_y_pinn = area * np.sum(pinn_solution_tri * midpoints_tri[:, 1]) / 3
                
                pinn_mass_total += local_mass_pinn
                pinn_moment_x += local_moment_x_pinn
                pinn_moment_y += local_moment_y_pinn
            
            # Compute centers of mass
            if crbe_mass_total > 1e-10:
                com_x_crbe = crbe_moment_x / crbe_mass_total
                com_y_crbe = crbe_moment_y / crbe_mass_total
            else:
                com_x_crbe, com_y_crbe = 0, 0
            
            if pinn_mass_total > 1e-10:
                com_x_pinn = pinn_moment_x / pinn_mass_total
                com_y_pinn = pinn_moment_y / pinn_mass_total
            else:
                com_x_pinn, com_y_pinn = 0, 0
            
            crbe_com_x.append(com_x_crbe)
            crbe_com_y.append(com_y_crbe)
            pinn_com_x.append(com_x_pinn)
            pinn_com_y.append(com_y_pinn)
        
        # Theoretical center of mass (for constant wind)
        theoretical_com_x = 10.0 + self.problem.v[0] * times  # Initial center at (10, 10)
        theoretical_com_y = 10.0 + self.problem.v[1] * times
        
        self.results['center_of_mass'] = {
            'times': times,
            'crbe_com_x': np.array(crbe_com_x),
            'crbe_com_y': np.array(crbe_com_y),
            'pinn_com_x': np.array(pinn_com_x),
            'pinn_com_y': np.array(pinn_com_y),
            'theoretical_com_x': theoretical_com_x,
            'theoretical_com_y': theoretical_com_y
        }
        
        return self.results['center_of_mass']
    
    def compute_spreading_rate_analysis(self):
        """Analyze the spreading rate (second moments) of the plume using triangle-based integration"""
        print("Computing spreading rate analysis...")
        
        times = self.mesh_data.time_discr
        crbe_var_x, crbe_var_y = [], []
        pinn_var_x, pinn_var_y = [], []
        
        for i, t in enumerate(times):
            # First pass: compute centers of mass
            crbe_mass_total = 0.0
            crbe_moment_x = 0.0
            crbe_moment_y = 0.0
            pinn_mass_total = 0.0
            pinn_moment_x = 0.0
            pinn_moment_y = 0.0
            
            for tri_idx in range(self.mesh_data.number_of_triangles):
                segs = self.mesh_data.triangle_to_segments[tri_idx]
                area = self.mesh_data.triangle_areas[tri_idx]
                midpoints_tri = self.mesh_data.midpoints[segs, :]
                
                # CRBE moments
                crbe_solution_tri = self.solver_crbe.solutions[i, segs]
                local_mass_crbe = area * np.sum(crbe_solution_tri) / 3
                crbe_mass_total += local_mass_crbe
                crbe_moment_x += area * np.sum(crbe_solution_tri * midpoints_tri[:, 0]) / 3
                crbe_moment_y += area * np.sum(crbe_solution_tri * midpoints_tri[:, 1]) / 3
                
                # PINN moments
                midpoints_tensor = torch.tensor(midpoints_tri, dtype=torch.float32, device=device)
                t_tensor = torch.full((len(segs), 1), t, dtype=torch.float32, device=device)
                xyt = torch.cat([midpoints_tensor, t_tensor], dim=1)
                
                with torch.no_grad():
                    pinn_solution_tri = self.model_pinn(xyt).squeeze().detach().cpu().numpy()
                
                local_mass_pinn = area * np.sum(pinn_solution_tri) / 3
                pinn_mass_total += local_mass_pinn
                pinn_moment_x += area * np.sum(pinn_solution_tri * midpoints_tri[:, 0]) / 3
                pinn_moment_y += area * np.sum(pinn_solution_tri * midpoints_tri[:, 1]) / 3
            
            # Centers of mass
            if crbe_mass_total > 1e-10:
                com_x_crbe = crbe_moment_x / crbe_mass_total
                com_y_crbe = crbe_moment_y / crbe_mass_total
            else:
                com_x_crbe, com_y_crbe = 0, 0
                
            if pinn_mass_total > 1e-10:
                com_x_pinn = pinn_moment_x / pinn_mass_total
                com_y_pinn = pinn_moment_y / pinn_mass_total
            else:
                com_x_pinn, com_y_pinn = 0, 0
            
            # Second pass: compute variances
            crbe_var_x_acc = 0.0
            crbe_var_y_acc = 0.0
            pinn_var_x_acc = 0.0
            pinn_var_y_acc = 0.0
            
            for tri_idx in range(self.mesh_data.number_of_triangles):
                segs = self.mesh_data.triangle_to_segments[tri_idx]
                area = self.mesh_data.triangle_areas[tri_idx]
                midpoints_tri = self.mesh_data.midpoints[segs, :]
                
                # CRBE variance
                crbe_solution_tri = self.solver_crbe.solutions[i, segs]
                crbe_var_x_acc += area * np.sum(crbe_solution_tri * (midpoints_tri[:, 0] - com_x_crbe)**2) / 3
                crbe_var_y_acc += area * np.sum(crbe_solution_tri * (midpoints_tri[:, 1] - com_y_crbe)**2) / 3
                
                # PINN variance
                midpoints_tensor = torch.tensor(midpoints_tri, dtype=torch.float32, device=device)
                t_tensor = torch.full((len(segs), 1), t, dtype=torch.float32, device=device)
                xyt = torch.cat([midpoints_tensor, t_tensor], dim=1)
                
                with torch.no_grad():
                    pinn_solution_tri = self.model_pinn(xyt).squeeze().detach().cpu().numpy()
                
                pinn_var_x_acc += area * np.sum(pinn_solution_tri * (midpoints_tri[:, 0] - com_x_pinn)**2) / 3
                pinn_var_y_acc += area * np.sum(pinn_solution_tri * (midpoints_tri[:, 1] - com_y_pinn)**2) / 3
            
            # Compute final variances
            if crbe_mass_total > 1e-10:
                var_x_crbe = crbe_var_x_acc / crbe_mass_total
                var_y_crbe = crbe_var_y_acc / crbe_mass_total
            else:
                var_x_crbe, var_y_crbe = 0, 0
                
            if pinn_mass_total > 1e-10:
                var_x_pinn = pinn_var_x_acc / pinn_mass_total
                var_y_pinn = pinn_var_y_acc / pinn_mass_total
            else:
                var_x_pinn, var_y_pinn = 0, 0
            
            crbe_var_x.append(var_x_crbe)
            crbe_var_y.append(var_y_crbe)
            pinn_var_x.append(var_x_pinn)
            pinn_var_y.append(var_y_pinn)
        
        # Theoretical spreading rate: σ²(t) = σ²₀ + 2Dt
        initial_variance = (12-8)**2 / 12  # For uniform distribution in [8,12]
        theoretical_var = initial_variance + 2 * self.problem.D * times
        
        self.results['spreading_rate'] = {
            'times': times,
            'crbe_var_x': np.array(crbe_var_x),
            'crbe_var_y': np.array(crbe_var_y),
            'pinn_var_x': np.array(pinn_var_x),
            'pinn_var_y': np.array(pinn_var_y),
            'theoretical_var': theoretical_var
        }
        
        return self.results['spreading_rate']
    
    def compute_peak_concentration_tracking(self):
        """Track peak concentration over time"""
        print("Computing peak concentration tracking...")
        
        times = self.mesh_data.time_discr
        midpoints = torch.tensor(self.mesh_data.midpoints, dtype=torch.float32, device=device)
        
        crbe_peaks = []
        pinn_peaks = []
        peak_locations_crbe = []
        peak_locations_pinn = []
        
        for i, t in enumerate(times):
            # CRBE peak
            crbe_solution = self.solver_crbe.solutions[i, :]
            crbe_peak_idx = np.argmax(crbe_solution)
            crbe_peak_val = crbe_solution[crbe_peak_idx]
            crbe_peak_loc = self.mesh_data.midpoints[crbe_peak_idx]
            
            crbe_peaks.append(crbe_peak_val)
            peak_locations_crbe.append(crbe_peak_loc)
            
            # PINN peak
            t_tensor = torch.full((midpoints.shape[0], 1), t, dtype=torch.float32, device=device)
            xyt = torch.cat([midpoints, t_tensor], dim=1)
            
            with torch.no_grad():
                pinn_solution = self.model_pinn(xyt).squeeze().detach().cpu().numpy()
            
            pinn_peak_idx = np.argmax(pinn_solution)
            pinn_peak_val = pinn_solution[pinn_peak_idx]
            pinn_peak_loc = self.mesh_data.midpoints[pinn_peak_idx]
            
            pinn_peaks.append(pinn_peak_val)
            peak_locations_pinn.append(pinn_peak_loc)
        
        self.results['peak_tracking'] = {
            'times': times,
            'crbe_peaks': np.array(crbe_peaks),
            'pinn_peaks': np.array(pinn_peaks),
            'crbe_peak_locations': np.array(peak_locations_crbe),
            'pinn_peak_locations': np.array(peak_locations_pinn)
        }
        
        return self.results['peak_tracking']
    
    def compute_concentration_profiles(self, y_slice=10.0):
        """Extract concentration profiles along specific transects"""
        print("Computing concentration profiles...")
        
        times = self.mesh_data.time_discr
        # Find points close to the y_slice
        midpoints = self.mesh_data.midpoints
        y_indices = np.where(np.abs(midpoints[:, 1] - y_slice) < 0.5)[0]  # Within 0.5 units
        
        selected_points = midpoints[y_indices]
        x_coords = selected_points[:, 0]
        sort_indices = np.argsort(x_coords)
        y_indices = y_indices[sort_indices]
        x_coords = x_coords[sort_indices]
        
        time_snapshots = [len(times)//4, len(times)//2, 3*len(times)//4, len(times)-1]
        
        profiles = {}
        for t_idx in time_snapshots:
            t = times[t_idx]
            crbe_profile = self.solver_crbe.solutions[t_idx, y_indices]
            
            # PINN profile
            midpoints_tensor = torch.tensor(selected_points[sort_indices], dtype=torch.float32, device=device)
            t_tensor = torch.full((len(y_indices), 1), t, dtype=torch.float32, device=device)
            xyt = torch.cat([midpoints_tensor, t_tensor], dim=1)
            
            with torch.no_grad():
                pinn_profile = self.model_pinn(xyt).squeeze().detach().cpu().numpy()
            
            profiles[f't_{t:.1f}'] = {
                'x_coords': x_coords,
                'crbe_profile': crbe_profile,
                'pinn_profile': pinn_profile
            }
        
        self.results['concentration_profiles'] = profiles
        return profiles
    
    def run_all_analyses(self):
        """Run all analyses"""
        print("Starting comprehensive analysis...")
        
        self.compute_mass_conservation()
        self.compute_center_of_mass_tracking()
        self.compute_spreading_rate_analysis()
        self.compute_peak_concentration_tracking()
        self.compute_concentration_profiles()
        
        print("All analyses completed!")
        return self.results
    
    def plot_all_results(self, save_dir="analysis_plots"):
        """Generate comprehensive plots for all analyses"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Set up plotting parameters
        plt.style.use('default')
        colors = {'crbe': '#1f77b4', 'pinn': '#ff7f0e', 'theoretical': '#2ca02c'}
        
        # 1. Mass conservation plot
        if 'mass_conservation' in self.results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            mc = self.results['mass_conservation']
            
            ax.plot(mc['times'], mc['crbe_masses'], 'o-', color=colors['crbe'], 
                   label='CRBE', markersize=4)
            ax.plot(mc['times'], mc['pinn_masses'], 's-', color=colors['pinn'], 
                   label='PINN', markersize=4)
            ax.axhline(y=mc['initial_mass'], color=colors['theoretical'], 
                      linestyle='--', label='Initial Mass')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Total Mass')
            ax.set_title('Mass Conservation Comparison')
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/mass_conservation.png", dpi=300)
            plt.savefig(f"{save_dir}/mass_conservation.pdf", dpi=600, bbox_inches='tight')
            plt.close()
        
        # 2. Center of mass tracking
        if 'center_of_mass' in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            com = self.results['center_of_mass']
            
            # X-direction
            ax1.plot(com['times'], com['crbe_com_x'], 'o-', color=colors['crbe'], 
                    label='CRBE', markersize=4)
            ax1.plot(com['times'], com['pinn_com_x'], 's-', color=colors['pinn'], 
                    label='PINN', markersize=4)
            ax1.plot(com['times'], com['theoretical_com_x'], '--', color=colors['theoretical'], 
                    label='Theoretical')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Center of Mass X (m)')
            ax1.set_title('Center of Mass - X Direction')
            ax1.legend(frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3)
            
            # Y-direction
            ax2.plot(com['times'], com['crbe_com_y'], 'o-', color=colors['crbe'], 
                    label='CRBE', markersize=4)
            ax2.plot(com['times'], com['pinn_com_y'], 's-', color=colors['pinn'], 
                    label='PINN', markersize=4)
            ax2.plot(com['times'], com['theoretical_com_y'], '--', color=colors['theoretical'], 
                    label='Theoretical')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Center of Mass Y (m)')
            ax2.set_title('Center of Mass - Y Direction')
            ax2.legend(frameon=True, fancybox=True, shadow=True)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/center_of_mass.png", dpi=300)
            plt.savefig(f"{save_dir}/center_of_mass.pdf", dpi=600, bbox_inches='tight')
            plt.close()
        
        # 3. Spreading rate analysis
        if 'spreading_rate' in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            sr = self.results['spreading_rate']
            
            # X-variance
            ax1.plot(sr['times'], sr['crbe_var_x'], 'o-', color=colors['crbe'], 
                    label='CRBE', markersize=4)
            ax1.plot(sr['times'], sr['pinn_var_x'], 's-', color=colors['pinn'], 
                    label='PINN', markersize=4)
            ax1.plot(sr['times'], sr['theoretical_var'], '--', color=colors['theoretical'], 
                    label='Theoretical')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Variance X (m²)')
            ax1.set_title('Plume Spreading - X Direction')
            ax1.legend(frameon=True, fancybox=True, shadow=True)
            ax1.grid(True, alpha=0.3)
            
            # Y-variance
            ax2.plot(sr['times'], sr['crbe_var_y'], 'o-', color=colors['crbe'], 
                    label='CRBE', markersize=4)
            ax2.plot(sr['times'], sr['pinn_var_y'], 's-', color=colors['pinn'], 
                    label='PINN', markersize=4)
            ax2.plot(sr['times'], sr['theoretical_var'], '--', color=colors['theoretical'], 
                    label='Theoretical')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Variance Y (m²)')
            ax2.set_title('Plume Spreading - Y Direction')
            ax2.legend(frameon=True, fancybox=True, shadow=True)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/spreading_rate.png", dpi=300)
            plt.savefig(f"{save_dir}/spreading_rate.pdf", dpi=600, bbox_inches='tight')
            plt.close()
        
        # 4. Peak concentration tracking
        if 'peak_tracking' in self.results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            pt = self.results['peak_tracking']
            
            ax.plot(pt['times'], pt['crbe_peaks'], 'o-', color=colors['crbe'], 
                   label='CRBE', markersize=4)
            ax.plot(pt['times'], pt['pinn_peaks'], 's-', color=colors['pinn'], 
                   label='PINN', markersize=4)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Peak Concentration')
            ax.set_title('Peak Concentration Evolution')
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/peak_concentration.png", dpi=300)
            plt.savefig(f"{save_dir}/peak_concentration.pdf", dpi=600, bbox_inches='tight')
            plt.close()
        
        # 5. Concentration profiles
        if 'concentration_profiles' in self.results:
            profiles = self.results['concentration_profiles']
            n_profiles = len(profiles)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, (time_key, profile_data) in enumerate(profiles.items()):
                if i < 4:  # Only plot first 4 time snapshots
                    ax = axes[i]
                    ax.plot(profile_data['x_coords'], profile_data['crbe_profile'], 
                           'o-', color=colors['crbe'], label='CRBE', markersize=4)
                    ax.plot(profile_data['x_coords'], profile_data['pinn_profile'], 
                           's-', color=colors['pinn'], label='PINN', markersize=4)
                    ax.set_xlabel('X coordinate (m)')
                    ax.set_ylabel('Concentration')
                    ax.set_title(f'Concentration Profile at {time_key}')
                    ax.legend(frameon=True, fancybox=True, shadow=True)
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/concentration_profiles.png", dpi=300)
            plt.savefig(f"{save_dir}/concentration_profiles.pdf", dpi=600, bbox_inches='tight')
            plt.close()
        
        print(f"All plots saved to {save_dir}/")

# Enhanced main function with comprehensive analysis
if __name__ == "__main__":
    # Problem setup (same as original)
    problem = Problem()
    domain = crbe.Domain()
    
    # Create mesh
    d_size = 20
    m_size = 64
    n_steps = 128
    
    mesh_file = crbe.create_mesh(m_size, domain_size=d_size)
    mesh = meshio.read(mesh_file)
    mesh_data = crbe.MeshData(mesh, domain, nt=n_steps)
    
    # CRBE solver
    cr_element = crbe.ElementCR()
    solver1 = crbe.BESCRFEM(domain, problem, mesh_data, cr_element, 1)
    solutions1 = solver1.solve()
    
    # PINN setup
    n_col = round(mesh_data.number_of_segments / 1.4)
    n_ic = round(0.35 * n_col)
    n_bc = round(0.05 * n_col)
    batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_bc}
    
    lambda_weights = {'pde': 1, 'ic': 8.0, 'bc': 1.0}
    lr = 1e-3
    epochs = 3000
    layers = [3] + [30] * 3 + [1]
    
    model = pinn.PINN(layers, problem, domain).to(device)
    model.train(
        batch_sizes, 
        epochs, 
        lr, 
        lambda_weights, 
        early_stopping_patience=10, 
        early_stopping_min_delta=1e-6,
        restore_best_weights=True
    )
    
    # Original error computation
    midpoints = torch.tensor(mesh_data.midpoints, dtype=torch.float32, device=device)
    t_tensor = torch.full((midpoints.shape[0], 1), mesh_data.domain.T, dtype=torch.float32, device=device)
    xyt = torch.cat([midpoints, t_tensor], dim=1)
    
    with torch.no_grad():
        u_pinn_midpoints = model(xyt).squeeze()
    
    u_pinn_midpoints = u_pinn_midpoints.detach().cpu().numpy()
    u_crbe_midpoints = solver1.solutions[-1, :]
    
    error = np.abs(u_pinn_midpoints - u_crbe_midpoints)
    l2_error = np.linalg.norm(error)
    max_error = np.max(np.abs(error))
    
    print(f"Original L2 error: {l2_error}")
    print(f"Original Max error: {max_error}")
    
    # Comprehensive analysis
    print("\n=== Starting Comprehensive Analysis ===")
    analyzer = ComprehensiveAnalysis(problem, domain, mesh_data, solver1, model)
    results = analyzer.run_all_analyses()
    
    # Generate plots
    analyzer.plot_all_results("section5_analysis_plots")
    
    # Print summary statistics
    print("\n=== Analysis Summary ===")
    
    if 'mass_conservation' in results:
        mc = results['mass_conservation']
        mass_loss_crbe = (mc['crbe_masses'][-1] - mc['crbe_masses'][0]) / mc['crbe_masses'][0] * 100
        mass_loss_pinn = (mc['pinn_masses'][-1] - mc['pinn_masses'][0]) / mc['pinn_masses'][0] * 100
        print(f"Mass conservation - CRBE loss: {mass_loss_crbe:.2f}%, PINN loss: {mass_loss_pinn:.2f}%")
    
    if 'center_of_mass' in results:
        com = results['center_of_mass']
        final_error_x_crbe = abs(com['crbe_com_x'][-1] - com['theoretical_com_x'][-1])
        final_error_x_pinn = abs(com['pinn_com_x'][-1] - com['theoretical_com_x'][-1])
        print(f"Center of mass error (final) - CRBE: {final_error_x_crbe:.2f}m, PINN: {final_error_x_pinn:.2f}m")
    
    if 'peak_tracking' in results:
        pt = results['peak_tracking']
        peak_decay_crbe = (pt['crbe_peaks'][0] - pt['crbe_peaks'][-1]) / pt['crbe_peaks'][0] * 100
        peak_decay_pinn = (pt['pinn_peaks'][0] - pt['pinn_peaks'][-1]) / pt['pinn_peaks'][0] * 100
        print(f"Peak concentration decay - CRBE: {peak_decay_crbe:.1f}%, PINN: {peak_decay_pinn:.1f}%")
        
    time_indices = [0, 64, n_steps-1]
    for it in time_indices:
        solver1.plot_interpolated_solution(time_index=it,name="crbe3")
        t = mesh_data.time_discr[it]
        _ = model.plot_interpolated_solution(t, mesh_data, name="pinn3")
        
