# %%writefile pinn.py
#v2.2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation
import time
from tqdm import tqdm
from pyDOE import lhs  # Latin Hypercube Sampling
import abc
import os 

# Set random seed for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

save_dir = "results"
os.makedirs(save_dir, exist_ok=True)
# ---------------------------------------------

# Sine activation 
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

#self.activation = Sine()

# Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# self.activation = Swish()

class AdaptiveTanh(nn.Module):
    def __init__(self, size):
        """One adaptive parameter (alpha) per neuron in the layer"""
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(size))  # Start with α=1 for each neuron

    def forward(self, x):
        return torch.tanh(self.alpha * x)


def backend(x):
    if isinstance(x, np.ndarray):
        return np
    elif isinstance(x, torch.Tensor):
        return torch
    else:
        raise TypeError("Unsupported type")
#Class de base pour les problêmes d'advection-diffusion avec params constants

class AdDifProblem(abc.ABC):
    def __init__(self, v, D):
        self.v = v
        self.D = D
        
    def initial_condition_fn(self, xyt):
        pass
    
    def boundary_fn(self, xyt):
        pass
    
    def source_term(self, xyt):
        pass
        
class Problem(AdDifProblem):
    """Physical model definitions and analytical solution."""
    
    def __init__(self, v=[1.0, 0.5], D=0.1, sigma=0.1):
        super().__init__(v, D)
        """Initialize model parameters."""
        self.sigma = sigma

    def analytical_solution(self, xyt):
        """Compute analytical solution at space-time points."""
        xp = backend(xyt)
        denom = 4 * self.D * xyt[:, 2] + self.sigma**2
        
        num = (xyt[:, 0] - self.v[0] * xyt[:, 2])**2 + (xyt[:, 1] - self.v[1] * xyt[:, 2])**2
        return xp.exp(- num /denom) / (xp.pi * denom)

    def initial_condition_fn(self, xy):
        """Evaluate initial condition."""
        xp = backend(xy)
        if xp == np:
            t = xp.zeros((xy.shape[0], 1), dtype=xp.float32)
            xyt = xp.hstack([xy, t])
        else:
            t = xp.zeros((xy.shape[0], 1), dtype=xp.float32, device=xy.device)
            xyt = xp.cat([xy, t], dim=1)
        
        return self.analytical_solution(xyt)

    def boundary_fn(self, xyt):
        return self.analytical_solution(xyt)
    
    def source_term(self, xyt):
        xp = backend(xyt)
        return xp.zeros_like(xyt[:,0])

class Domain:
    """Parameters defining the domain of the problem."""
    
    def __init__(self, Lx=20, Ly=20, T=10):
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
     

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-6, restore_best_weights=True):
        """Early stopping utility to stop training when loss stops improving."""
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """Check if training should stop and update best weights."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_weights(self, model):
        """Restore the best weights to the model"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
   

# Define the PINN model
class PINN(nn.Module):
    def __init__(self, layers, problem, domain, activation="adaptive_tanh"):
        super(PINN, self).__init__()
        self.problem = problem
        self.domain = domain
        self.xy_ranges = [-domain.Lx, domain.Lx, -domain.Ly, domain.Ly]
        self.t_range = [0, domain.T]
        
        # Build the neural network
            
        self.loss_function = nn.MSELoss(reduction='mean')
        
        # Create layers
        layer_list = []
        for i in range(len(layers)-2):
            layer = nn.Linear(layers[i], layers[i+1])
            layer_list.append(layer)
            # Weight initialization
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            #Adding Activation
            if activation == "adaptive_tanh":
            	layer_list.append(AdaptiveTanh(layers[i+1]))
            elif activation == "tanh":
            	layer_list.append(nn.Tanh())
            elif activation == "sine":
            	layer_list.append(Sine())
            elif activation == "swish":
            	layer_list.append(Swish())
            else:
            	raise ValueError(f"Activation function {activation} not implemented")
        
        layer_list.append(nn.Linear(layers[-2], layers[-1]))
        self.model = nn.Sequential(*layer_list)
        
        # Move model to device
        self.to(device)
        
    def forward(self, xyt):
        """Forward pass through the network"""
        return self.model(xyt)
    
    def compute_pde_residual(self, xyt):
        """Compute the PDE residual: dc/dt + v·∇c - D·∆c - f"""
        D = xyt.shape[1]
        #xyt.requires_grad_(True)
        xyt = xyt.clone().detach().requires_grad_(True)
        
        c = self.forward(xyt)
        grad_xy, grad_t, laplacian_xy = compute_gradient_and_laplacian_xy(c, xyt)
        
        #v_dot_grad = sum([torch.tensor(self.problem.v)[d] * grad_xy[:, d:d+1] for d in range(D-1)])
        v = torch.tensor(self.problem.v, device=xyt.device)
        v_dot_grad = torch.sum(v[:D-1] * grad_xy, dim=1, keepdim=True)
        source = self.problem.source_term(xyt).to(xyt.device).unsqueeze(-1)
        
        
        return grad_t + v_dot_grad - self.problem.D * laplacian_xy - source
    
    def train(self, batch_sizes, epochs, lr, lambda_weights, early_stopping_patience=500, early_stopping_min_delta=1e-6, mini_batch_size=None, restore_best_weights=True):
        """Train the PINNusing Latin Hypercube sampling for collocation points"""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, verbose=True)
        
        #initialier early stopping
        early_stopping = EarlyStopping(
            patience=early_stopping_patience, 
            min_delta=early_stopping_min_delta,
            restore_best_weights=restore_best_weights
        )
        
        # Create history to track losses
        self.history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': []}
        
        start_time = time.time()
        
        #Pre-generate lhs points for initial condition
        xy_ic = lhs_sampling(batch_sizes['ic'], self.xy_ranges)
        t_ic = torch.zeros((batch_sizes['ic'], 1), device=device)
        xyt_ic = torch.cat([xy_ic, t_ic], dim=1).to(device)
        
        #training loop
        for epoch in tqdm(range(epochs)):
            
            xyt_bc = sample_boundary_points(batch_sizes['bc'], self.xy_ranges, self.t_range) #collocation for bc
            
            # zero out gradient
            optimizer.zero_grad()
            
            xyt = lhs_sampling(batch_sizes['pde'], self.xy_ranges, self.t_range)  # collocation for pde

            if batch_sizes['pde'] > 4096:
                mini_batch_size = mini_batch_size or 4096
                n_points = xyt.shape[0]
                
                pde_loss = 0.0
                
                losses = []
                for i in range(0, n_points, mini_batch_size):
                	xyt_mini = xyt[i:i+mini_batch_size]
                	residual = self.compute_pde_residual(xyt_mini)
                	losses.append(torch.mean(torch.square(residual)))
                pde_loss = torch.mean(torch.stack(losses))
            else:
                pde_loss = torch.mean(
                    torch.square(self.compute_pde_residual(xyt))
                )

            
            ic_loss = self.loss_function(
                self.forward(xyt_ic),
                self.problem.initial_condition_fn(xy_ic).reshape(-1, 1)
            )
            
            bc_loss = self.loss_function(
                self.forward(xyt_bc),
                self.problem.boundary_fn(xyt_bc).reshape(-1, 1)
            )
            
            # Total loss
            total_lambda_weight = lambda_weights['pde'] + lambda_weights['ic'] + lambda_weights['bc']
            total_loss = (
                lambda_weights['pde'] * pde_loss + 
                lambda_weights['ic'] * ic_loss + 
                lambda_weights['bc'] * bc_loss
            ) / total_lambda_weight
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            
            # Update learning rate
            scheduler.step(total_loss)
            
            #recordlosses
            self.history['total_loss'].append(total_loss.item())
            self.history['pde_loss'].append(pde_loss.item())
            self.history['ic_loss'].append(ic_loss.item())
            self.history['bc_loss'].append(bc_loss.item())
            
            #check early stopping
            if early_stopping(total_loss.item(), self.model):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best loss: {early_stopping.best_loss:.6f}")
                break
        #restore best weights if early stopping was used
        if early_stopping.restore_best_weights:
            early_stopping.restore_weights(self.model)
            print("Restored best model weights")
            
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return self.history

    def compute_errors(self, mesh_data, analytical_sol_fn):
            """Compute errors between numerical and analytical solutions on GPU."""
            rel_l2_error = torch.tensor(0.0, dtype=torch.float32, device=device)
            max_error = torch.tensor(0.0, dtype=torch.float32, device=device)
            l2_error = torch.tensor(0.0, dtype=torch.float32, device=device)
            
            _norm_u_exact = torch.tensor(0.0, dtype=torch.float32, device=device)
            
            midpoints = torch.tensor(mesh_data.midpoints, dtype=torch.float32, device=device)
            t_tensor = torch.full((midpoints.shape[0], 1), self.domain.T, dtype=torch.float32, device=device)
            midpoints_t = torch.cat([midpoints, t_tensor], dim=1)
            
            for tri_idx in range(mesh_data.number_of_triangles):
                segs = mesh_data.triangle_to_segments[tri_idx]
                
                xyt = midpoints_t[segs,:]

                with torch.no_grad():
                    u_exact_midpoints = analytical_sol_fn(xyt).squeeze()
                    u_num_midpoints = self.forward(xyt).squeeze()

                area = mesh_data.triangle_areas[tri_idx]
                local_error = area * torch.sum((u_num_midpoints - u_exact_midpoints) ** 2)
                local_norm_u_exact = area * torch.sum(u_exact_midpoints ** 2)

                l2_error += local_error
                _norm_u_exact += local_norm_u_exact
                
                # compute pointwise max error on this triangle
                local_max_error = torch.max(torch.abs(u_num_midpoints - u_exact_midpoints))
                max_error = torch.maximum(max_error, local_max_error)
                
                #max_error = torch.maximum(max_error, local_error)
                
            _norm_u_exact /= 3
            l2_error /= 3
            #max_error /= 3

            rel_l2_error = l2_error / (_norm_u_exact + 1e-12)  # avoid division by zero

            return rel_l2_error.item(), l2_error.item(), max_error.item()

    def plot_history(self, save_dir="results", name=""):
        """Plot training loss history"""
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.history['total_loss'], label='Total Loss', ls="-.")
        plt.semilogy(self.history['pde_loss'], label='PDE Loss')
        plt.semilogy(self.history['ic_loss'], label='IC Loss')
        plt.semilogy(self.history['bc_loss'], label='BC Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(f"{save_dir}/loss_history_{name}.pdf", dpi=500)
        plt.savefig(f"{save_dir}/loss_history_{name}.png", dpi=500)
        plt.tight_layout()
        plt.close()
    
    def plot_solution(self, t, mesh_data, analytical_sol_fn=None, save_dir="results"):
        """Plot error evolution over time."""
        
        points = torch.tensor(mesh_data.points[:, 0:2], dtype=torch.float32, device=device)
        triangles = mesh_data.triangles
        t_tensor = t * torch.ones_like(points[:, 0:1])
        xyt = torch.cat([points, t_tensor], dim=1)
        
        points = points.cpu().numpy()
        
        with torch.no_grad():
            u_num = self.forward(xyt).cpu().numpy().flatten()
            
        if analytical_sol_fn:
            analytical_vertex_values = analytical_sol_fn(xyt).cpu().numpy().flatten()
            
            # Create subplot
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))

            # Plot numerical solution
            triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
            cntr1 = axs[0].tricontourf(triang, u_num, 20, cmap="viridis")
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
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
            cntr1 = ax.tricontourf(triang, u_num, 20, cmap="viridis")
            ax.set_title(f"Numerical Solution at t = {t:.3f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(cntr1, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/solution_{t}.pdf", dpi=500)
        plt.savefig(f"{save_dir}/solution_{t}.png", dpi=500)
        plt.close()
        
        print(f"Saved at {save_dir}/solution_{t}.pdf-png")
        
    def plot_interpolated_solution(self, t, mesh_data, analytical_sol_fn=None, save_dir="results", name=""):
        """Plot error evolution over time."""
        os.makedirs(save_dir, exist_ok=True)
        
        #Evaluate the PINN at midpoint like CR
        midpoints = torch.tensor(mesh_data.midpoints, dtype=torch.float32, device=device)
        t_tensor = t * torch.ones_like(midpoints[:, 0:1], device=device)
        xyt = torch.cat([midpoints, t_tensor], dim=1)

        midpoints = midpoints.cpu().numpy()
        
        with torch.no_grad():
            u_num = self.forward(xyt).cpu().numpy().flatten()
            
        points = torch.tensor(mesh_data.points[:, 0:2], dtype=torch.float32, device=device)
        triangles = mesh_data.triangles
        
        t_tensor = t * torch.ones_like(points[:, 0:1], device=device)
        xyt = torch.cat([points, t_tensor], dim=1)

        points = points.cpu().numpy()
        
        #Then interpolate to evaluate at triangles nodes
        vertex_values=  np.zeros(len(points))
        count = np.zeros(len(points))
        
        for i, (a, b) in enumerate(mesh_data.segments):
            vertex_values[a] += u_num[i]
            vertex_values[b] += u_num[i]
            count[a] += 1
            count[b] += 1
        
        #Average at vertices
        vertex_values /= np.maximum(count, 1)
        
        if analytical_sol_fn:
            analytical_vertex_values = analytical_sol_fn(xyt).cpu().numpy().flatten()
            
            # Create subplot
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))

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
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
            cntr1 = ax.tricontourf(triang, vertex_values, 20, cmap="viridis")
            ax.set_title(f"Numerical Solution at t = {t:.3f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(cntr1, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pinn_interpolated_solution_{name}.pdf", dpi=500)
        plt.savefig(f"{save_dir}/pinn_interpolated_solution_{name}.png", dpi=500)
        plt.close()
        
        print(f"Saved at {save_dir}/pinn_interpolated_solution_{name}.pdf-png")




def compute_gradient_and_laplacian_xy(model, xyt):
    D = xyt.shape[1]

    grad_c = torch.autograd.grad(model, 
        		xyt,
        		grad_outputs=torch.ones_like(model),
        		retain_graph=True,
        		create_graph=True,
        )[0]
    
    grad_xy = grad_c[:, :D-1]
    
    grad_t = grad_c[:, D-1:D]
    
    def second_derivative(grad_component, dim):
        """Compute second derivative w.r.t. a single dim"""
        return torch.autograd.grad(
            grad_component,
            xyt,
            grad_outputs=torch.ones_like(grad_component),
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0][:, dim]
    
    laplacian_xy = sum([second_derivative(grad_c[d:d+1], dim=d) for d in range(D-1)])
    
    return grad_xy, grad_t, laplacian_xy


def lhs_sampling(n_samples, domain, time_range=None):
    """
    Generate Latin Hypercube samples as a single tensor with shape (n_samples, D)
    """
    x_min, x_max, y_min, y_max = domain

    if time_range is None:
        samples = lhs(2, n_samples)
        x_samples = (x_max - x_min) * samples[:, 0] + x_min
        y_samples = (y_max - y_min) * samples[:, 1] + y_min
        data = torch.tensor(np.stack([x_samples, y_samples], axis=1), dtype=torch.float32, device=device)
    else:
        t_min, t_max = time_range
        samples = lhs(3, n_samples)
        t_samples = (t_max - t_min) * samples[:, 0] + t_min
        x_samples = (x_max - x_min) * samples[:, 1] + x_min
        y_samples = (y_max - y_min) * samples[:, 2] + y_min
        data = torch.tensor(np.stack([x_samples, y_samples, t_samples], axis=1), dtype=torch.float32, device=device)
    return data

def sample_boundary_points(n_samples, domain, time_range):
    """Sample points from the boundary using Latin Hypercube Sampling"""
    x_min, x_max, y_min, y_max = domain
    t_min, t_max = time_range

    # Number of points per boundary
    n_per_boundary = n_samples // 4

    # Sample times for all boundaries
    t_samples = lhs(1, n_samples)
    t_bc = torch.tensor((t_max - t_min) * t_samples + t_min, dtype=torch.float32, device=device)

    # Left boundary (x = x_min)
    y_bc_left = torch.tensor((y_max - y_min) * lhs(1, n_per_boundary) + y_min, dtype=torch.float32, device=device)
    x_bc_left = torch.full_like(y_bc_left, x_min)

    # Right boundary (x = x_max)
    y_bc_right = torch.tensor((y_max - y_min) * lhs(1, n_per_boundary) + y_min, dtype=torch.float32, device=device)
    x_bc_right = torch.full_like(y_bc_right, x_max)

    # Bottom boundary (y = y_min)
    x_bc_bottom = torch.tensor((x_max - x_min) * lhs(1, n_per_boundary) + x_min, dtype=torch.float32, device=device)
    y_bc_bottom = torch.full_like(x_bc_bottom, y_min)

    # Top boundary (y = y_max)
    x_bc_top = torch.tensor((x_max - x_min) * lhs(1, n_per_boundary) + x_min, dtype=torch.float32, device=device)
    y_bc_top = torch.full_like(x_bc_top, y_max)

    # Combine boundary points
    x_bc = torch.cat([x_bc_left, x_bc_right, x_bc_bottom, x_bc_top], dim=0)
    y_bc = torch.cat([y_bc_left, y_bc_right, y_bc_bottom, y_bc_top], dim=0)
    t_bc = t_bc[:x_bc.shape[0]]

    # Stack as [x, y, t]
    xyt_bc = torch.cat([x_bc, y_bc, t_bc], dim=1)  # All are (N, 1) tensors

    return xyt_bc


print("Loading pinn.py")

if __name__ == "__main__":
    print("Running main block in pinn.py")

    #  Initialize
    domain = Domain()
    
    problem = Problem()
    
    #mmeshing
    import crbe
    import meshio
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    
    mesh_file = crbe.create_mesh(64, domain_size=20.0)
    mesh = meshio.read(mesh_file)
    mesh_data = crbe.MeshData(mesh, domain, nt=128)
    
    # PINN's setup
    layers = [3] + [32] * 4 + [1]  # Input: (x, y, t) → Output: c(x, y, t)
    pinn = PINN(layers, problem, domain).to(device)

    #xyt = torch.tensor([[1.0, 0.1, 0.0]], device=device, requires_grad=True)  # Shape: (1, 3)

    # Compute residual
    #residual = pinn.compute_pde_residual(xyt)
    #print("PDE residual at (1.0, 0.1, 0.0):", residual)
    
    n_ic = round(0.2 * mesh_data.number_of_segments)
    n_bc = n_ic
    n_col = mesh_data.number_of_segments - n_ic - n_bc
    batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_ic}
    lambda_weights = {'pde': 180.0, 'ic': 80.0, 'bc': 80.0}#60, 40, 100); (150, 40, 80); (180, 60, 80)
    
    lr = 1e-4
    epochs = 4#000
    
    pinn.train(batch_sizes, epochs, lr, lambda_weights, early_stopping_patience=1000, early_stopping_min_delta=1e-6)
    
    #xy = torch.tensor([[1.5, 3], [5, 4]], device=device, requires_grad=True)
    
    pinn.plot_history()
    
    errors = pinn.compute_errors(mesh_data, problem.analytical_solution)
    print(f"Compute error\n\tRel L2 Error: {errors[0]:.4f}\n\tL2 Error: {errors[1]:.4f}\n\tMax Error: {errors[2]:.4f}")
    print()
    
    pinn.plot_interpolated_solution(10.0, mesh_data, problem.analytical_solution)
    
    #for name, param in pinn.named_parameters():
    #	if "alpha" in name:
    #		print(name, param.data)

