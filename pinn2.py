# %%writefile pinn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from tqdm import tqdm
from pyDOE import lhs  # Latin Hypercube Sampling

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



# Set random seed for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PINN(nn.Module):
    def __init__(self, layers, D, v, activation="tanh"):
        """
        Initialize the Physics-Informed Neural Network
        
        Args:
            layers: List of integers, number of neurons in each layer
            D: Diffusion coefficient
            v: Velocity vector [v_x, v_y]
        """
        super(PINN, self).__init__()
        self.D = D
        self.v = v
        self.sigma = 1.0  # Initial condition parameter
        
        # Build the neural network
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sine":
            self.activation = Sine()
        else:
            self.activation = Swish()
            
        self.loss_function = nn.MSELoss(reduction='mean')
        
        # Create layers
        layer_list = []
        for i in range(len(layers)-2):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            # Weight initialization
            nn.init.xavier_normal_(layer_list[-1].weight)
            nn.init.zeros_(layer_list[-1].bias)
            #Adding Activation
            layer_list.append(self.activation)
        layer_list.append(nn.Linear(layers[-2], layers[-1]))
        
        self.model = nn.Sequential(*layer_list)
        
        # Move model to device
        self.to(device)
    
    def forward(self, t, x, y):
        """Forward pass through the network"""
        # Combine inputs
        inputs = torch.cat([t, x, y], dim=1)
        return self.model(inputs)
    
    def f_analytical(self, t, x, y):
        """Analytical solution for the pollutant concentration"""
        # Move to CPU for numpy operations
        t_np = t.cpu().detach().numpy()
        x_np = x.cpu().detach().numpy()
        y_np = y.cpu().detach().numpy()
        
        v_x, v_y = self.v
        denominator = np.pi * (4 * self.D * t_np + self.sigma**2)
        numerator = np.exp(-((x_np - v_x * t_np)**2 + (y_np - v_y * t_np)**2) / (4 * self.D * t_np + self.sigma**2))
        result = numerator / denominator
        # Return result to the same device as input tensors
        return torch.tensor(result, dtype=torch.float32, device=t.device)
    
    def compute_pde_residual(self, t, x, y):
        """Compute the PDE residual: dc/dt + v·∇c - D·∆c"""
        t.requires_grad_(True)
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        c = self.forward(t, x, y)
        
        # Compute derivatives using automatic differentiation
        c_t = torch.autograd.grad(
            c, t, 
            grad_outputs=torch.ones_like(c),
            retain_graph=True,
            create_graph=True
        )[0]
        
        c_x = torch.autograd.grad(
            c, x, 
            grad_outputs=torch.ones_like(c),
            retain_graph=True,
            create_graph=True
        )[0]
        
        c_y = torch.autograd.grad(
            c, y, 
            grad_outputs=torch.ones_like(c),
            retain_graph=True,
            create_graph=True
        )[0]
        
        # Compute second derivatives with check for unused gradients
        c_xx = torch.autograd.grad(
            c_x, x, 
            grad_outputs=torch.ones_like(c_x),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        
        c_yy = torch.autograd.grad(
            c_y, y, 
            grad_outputs=torch.ones_like(c_y),
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        
        # Check for None values and set them to zero if unused
        if c_xx is None:
            c_xx = torch.zeros_like(c_x)
        if c_yy is None:
            c_yy = torch.zeros_like(c_y)
        
        # Compute PDE residual
        v_x, v_y = self.v
        return c_t + v_x * c_x + v_y * c_y - self.D * (c_xx + c_yy)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=100, min_delta=1e-6, restore_best_weights=True):
        """
        Early stopping utility to stop training when loss stops improving.
        
        Args:
            patience: Number of epochs to wait before stopping after loss stops improving
            min_delta: Minimum change in loss to qualify as an improvement
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """
        Check if training should stop and update best weights.
        
        Args:
            val_loss: Current validation loss
            model: The model being trained
            
        Returns:
            True if training should stop, False otherwise
        """
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

def lhs_sampling(n_samples, domain, time_range=None):
    """
    Generate Latin Hypercube samples within the domain
    
    Args:
        n_samples: Number of samples to generate
        domain: Domain boundaries [x_min, x_max, y_min, y_max]
        time_range: Time range [t_min, t_max] or None for spatial sampling only
    
    Returns:
        Tuple of tensors for each dimension
    """
    x_min, x_max, y_min, y_max = domain
    
    # Determine sampling dimensionality (2D or 3D)
    if time_range is None:
        # Generate 2D Latin Hypercube samples
        samples = lhs(2, n_samples)
        # Scale to domain range
        x_samples = (x_max - x_min) * samples[:, 0:1] + x_min
        y_samples = (y_max - y_min) * samples[:, 1:2] + y_min
        return (torch.tensor(x_samples, dtype=torch.float32, device=device), 
                torch.tensor(y_samples, dtype=torch.float32, device=device))
    else:
        # Generate 3D Latin Hypercube samples
        t_min, t_max = time_range
        samples = lhs(3, n_samples)
        # Scale to domain range
        t_samples = (t_max - t_min) * samples[:, 0:1] + t_min
        x_samples = (x_max - x_min) * samples[:, 1:2] + x_min
        y_samples = (y_max - y_min) * samples[:, 2:3] + y_min
        return (torch.tensor(t_samples, dtype=torch.float32, device=device), 
                torch.tensor(x_samples, dtype=torch.float32, device=device), 
                torch.tensor(y_samples, dtype=torch.float32, device=device))


def sample_boundary_points(n_samples, domain, time_range):
    """
    Sample points from the boundary using Latin Hypercube Sampling
    
    Args:
        n_samples: Number of boundary points to sample
        domain: Domain boundaries [x_min, x_max, y_min, y_max]
        time_range: Time range [t_min, t_max]
    
    Returns:
        Tuple of tensors (t_bc, x_bc, y_bc)
    """
    x_min, x_max, y_min, y_max = domain
    t_min, t_max = time_range
    
    # Number of points per boundary
    n_per_boundary = n_samples // 4
    
    # Sample times for all boundaries using Latin Hypercube
    t_samples = lhs(1, n_samples)
    t_bc = torch.tensor((t_max - t_min) * t_samples + t_min, dtype=torch.float32, device=device)
    
    # Left boundary (x = x_min)
    left_samples = lhs(1, n_per_boundary)
    y_bc_left = torch.tensor((y_max - y_min) * left_samples + y_min, dtype=torch.float32, device=device)
    x_bc_left = torch.ones_like(y_bc_left) * x_min
    
    # Right boundary (x = x_max)
    right_samples = lhs(1, n_per_boundary)
    y_bc_right = torch.tensor((y_max - y_min) * right_samples + y_min, dtype=torch.float32, device=device)
    x_bc_right = torch.ones_like(y_bc_right) * x_max
    
    # Bottom boundary (y = y_min)
    bottom_samples = lhs(1, n_per_boundary)
    x_bc_bottom = torch.tensor((x_max - x_min) * bottom_samples + x_min, dtype=torch.float32, device=device)
    y_bc_bottom = torch.ones_like(x_bc_bottom) * y_min
    
    # Top boundary (y = y_max)
    top_samples = lhs(1, n_per_boundary)
    x_bc_top = torch.tensor((x_max - x_min) * top_samples + x_min, dtype=torch.float32, device=device)
    y_bc_top = torch.ones_like(x_bc_top) * y_max
    
    # Combine all boundary points
    x_bc = torch.cat([x_bc_left, x_bc_right, x_bc_bottom, x_bc_top], 0)
    y_bc = torch.cat([y_bc_left, y_bc_right, y_bc_bottom, y_bc_top], 0)
    
    # Ensure we have the right number of time points
    t_bc = t_bc[:x_bc.size(0)]
    
    return t_bc, x_bc, y_bc


# def train_pinn(model, domain, time_range, batch_sizes, learning_rate, epochs, lambda_weights):
def train_pinn(model, domain, time_range, batch_sizes, learning_rate, epochs, lambda_weights, 
               early_stopping_patience=1000, early_stopping_min_delta=1e-6):
    """
    Train the PINN model using Latin Hypercube sampling for collocation points
    
    Args:
        model: The PINN model
        domain: Domain boundaries [x_min, x_max, y_min, y_max]
        time_range: Time range [t_min, t_max]
        batch_sizes: Dictionary with batch sizes for each type of points
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        lambda_weights: Dictionary with weights for each loss term
        early_stopping_patience: Number of epochs to wait before early stopping
        early_stopping_min_delta: Minimum change in loss for early stopping
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, verbose=True)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, 
        min_delta=early_stopping_min_delta,
        restore_best_weights=True
    )
    
    # Create history to track losses
    history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': []}
    
    start_time = time.time()
    
    # Pre-generate LHS points for initial condition
    x_ic, y_ic = lhs_sampling(batch_sizes['ic'], domain)
    t_ic = torch.zeros((batch_sizes['ic'], 1), device=device)
    
    # Training loop
    for epoch in tqdm(range(epochs)):
        # Sample collocation points for PDE residual using LHS
        t_pde, x_pde, y_pde = lhs_sampling(batch_sizes['pde'], domain, time_range)
        
        # Sample boundary points
        t_bc, x_bc, y_bc = sample_boundary_points(batch_sizes['bc'], domain, time_range)
        
        # Zero out gradients
        optimizer.zero_grad()
        
        # Compute losses
        # PDE residual loss
        pde_residual = model.compute_pde_residual(t_pde, x_pde, y_pde)
        pde_loss = torch.mean(torch.square(pde_residual))
        
        # Initial condition loss
        c_ic_pred = model(t_ic, x_ic, y_ic)
        c_ic_true = model.f_analytical(t_ic, x_ic, y_ic).reshape(-1, 1)
        ic_loss = model.loss_function(c_ic_pred, c_ic_true)
        
        # Boundary condition loss
        c_bc_pred = model(t_bc, x_bc, y_bc)
        c_bc_true = model.f_analytical(t_bc, x_bc, y_bc).reshape(-1, 1)
        bc_loss = model.loss_function(c_bc_pred, c_bc_true)
        
        # Total loss
        total_loss = (
            lambda_weights['pde'] * pde_loss + 
            lambda_weights['ic'] * ic_loss + 
            lambda_weights['bc'] * bc_loss
        )
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step(total_loss)
        
        # Record losses
        history['total_loss'].append(total_loss.item())
        history['pde_loss'].append(pde_loss.item())
        history['ic_loss'].append(ic_loss.item())
        history['bc_loss'].append(bc_loss.item())
        
        # Check early stopping
        if early_stopping(total_loss.item(), model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best loss: {early_stopping.best_loss:.6f}")
            break
    
    # Restore best weights if early stopping was used
    if early_stopping.restore_best_weights:
        early_stopping.restore_weights(model)
        print("Restored best model weights")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return history


def evaluate_model(model, domain, T_max, resolution=50):
    """
    Evaluate the trained model and compare with analytical solution
    
    Args:
        model: Trained PINN model
        domain: Domain boundaries [x_min, x_max, y_min, y_max]
        T_max: Maximum time
        resolution: Number of points in each dimension for visualization
    """
    x_min, x_max, y_min, y_max = domain
    
    # Create grid for visualization
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Time points to visualize
    time_points = [T_max] # [0.0, T_max/4, T_max/2, 3*T_max/4, T_max]
    
    errors = []
    
    fig, axs = plt.subplots(len(time_points), 2, figsize=(15, 5*len(time_points)))
    
    for i, t_val in enumerate(time_points):
        # Convert to torch tensors
        t = torch.tensor(np.full((resolution*resolution, 1), t_val), dtype=torch.float32, device=device)
        x_flat = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        y_flat = torch.tensor(Y.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
        
        # Model prediction
        with torch.no_grad():
            c_pred = model(t, x_flat, y_flat).cpu().numpy().reshape(resolution, resolution)
        
        # Analytical solution
        c_analytical = model.f_analytical(t, x_flat, y_flat).cpu().numpy().reshape(resolution, resolution)
        
        # # Compute error
        # error = np.abs(c_pred - c_analytical)
        # error_relative = np.divide(error, c_analytical + 1e-10)
        # error_mean = np.mean(error)
        # errors.append(error_mean)
        
        # Plot results
        axs[0].contourf(X, Y, c_pred, 50, cmap='jet')
        axs[0].set_title(f'PINN prediction at t = {t_val:.2f}')
        axs[0].set_xlabel('x (km)')
        axs[0].set_ylabel('y (km)')
        
        im = axs[1].contourf(X, Y, c_analytical, 50, cmap='jet')
        axs[1].set_title(f'Analytical solution at t = {t_val:.2f}')
        axs[1].set_xlabel('x (km)')
        axs[1].set_ylabel('y (km)')
        
        # axs[i, 2].contourf(X, Y, error_relative, 50, cmap='viridis')
        # axs[i, 2].set_title(f'Relative error at t = {t_val:.2f}, Mean: {error_mean:.6f}')
        # axs[i, 2].set_xlabel('x (km)')
        # axs[i, 2].set_ylabel('y (km)')
        
    
    fig.tight_layout()
    plt.colorbar(im, ax=axs.ravel().tolist())
    plt.savefig(f"solution_{t_val}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"solution_{t_val}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return errors


def plot_loss_history(history, f_name="loss_history.pdf"):
    """Plot training loss history"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(history['total_loss'], label='Total Loss')
    plt.semilogy(history['pde_loss'], label='PDE Loss')
    plt.semilogy(history['ic_loss'], label='IC Loss')
    plt.semilogy(history['bc_loss'], label='BC Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f_name, dpi=500)
    plt.tight_layout()
    plt.close()


def plot_collocation_points(domain, time_range, batch_sizes):
    """Visualize the distribution of collocation points from Latin Hypercube Sampling"""
    # Generate LHS points
    t_pde, x_pde, y_pde = lhs_sampling(batch_sizes['pde'], domain, time_range)
    x_ic, y_ic = lhs_sampling(batch_sizes['ic'], domain)
    t_bc, x_bc, y_bc = sample_boundary_points(batch_sizes['bc'], domain, time_range)
    
    # Convert to numpy for plotting (move to CPU first)
    x_pde_np = x_pde.cpu().numpy()
    y_pde_np = y_pde.cpu().numpy()
    t_pde_np = t_pde.cpu().numpy()
    
    x_ic_np = x_ic.cpu().numpy()
    y_ic_np = y_ic.cpu().numpy()
    
    x_bc_np = x_bc.cpu().numpy()
    y_bc_np = y_bc.cpu().numpy()
    
    # Plot spatial distribution of collocation points
    plt.figure(figsize=(10, 8))
    plt.scatter(x_pde_np, y_pde_np, s=2, alpha=0.5, label='PDE Residual Points')
    plt.scatter(x_ic_np, y_ic_np, s=5, color='red', label='Initial Condition Points')
    plt.scatter(x_bc_np, y_bc_np, s=5, color='green', label='Boundary Points')
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.title('Spatial Distribution of Collocation Points')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    # Plot space-time distribution of PDE residual points
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(t_pde_np, x_pde_np, y_pde_np, s=2, alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('x (km)')
    ax.set_zlabel('y (km)')
    ax.set_title('Space-Time Distribution of PDE Residual Points')
    plt.show()

def compute_error(model, t, x, y):
    """Calculate error between model prediction and exact solution"""
    # Move inputs to device
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
    x_flat = torch.tensor(x.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
    y_flat = torch.tensor(y.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
            
    # Model prediction
    with torch.no_grad():
        u_pred = model(t_tensor, x_flat, y_flat).cpu().numpy().flatten()
            
    # Analytical solution
    u_exact = model.f_analytical(t_tensor, x_flat, y_flat).cpu().numpy().flatten()

    # Calculate errors
    error = np.abs(u_pred - u_exact)
    rel_l2_error = np.sqrt(np.mean(np.square(error))) / (np.sqrt(np.mean(np.square(u_exact))) + 1e-10)
    max_error = np.max(error)

    return rel_l2_error, max_error, u_pred, u_exact

def compute_convergence_metrics(model, domain, time_range, resolutions=[20, 40, 60, 80]):
    """
    Compute convergence metrics by comparing with analytical solutions at different resolutions
    
    Args:
        model: Trained PINN model
        domain: Domain boundaries [x_min, x_max, y_min, y_max]
        time_range: Time range [t_min, t_max]
        resolutions: List of grid resolutions to evaluate
    """
    x_min, x_max, y_min, y_max = domain
    t_min, t_max = time_range
    
    # Time points to evaluate
    time_points = [0.0, t_max/4, t_max/2, 3*t_max/4, t_max]
    
    errors_l2 = []
    errors_linf = []
    
    for resolution in resolutions:
        # Create grid
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        res_errors_l2 = []
        res_errors_linf = []
        
        for t_val in time_points:
            # Convert to torch tensors on device
            t = torch.tensor(np.full((resolution*resolution, 1), t_val), dtype=torch.float32, device=device)
            x_flat = torch.tensor(X.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
            y_flat = torch.tensor(Y.flatten().reshape(-1, 1), dtype=torch.float32, device=device)
            
            # Model prediction
            with torch.no_grad():
                c_pred = model(t, x_flat, y_flat).cpu().numpy().flatten()
            
            # Analytical solution
            c_analytical = model.f_analytical(t, x_flat, y_flat).cpu().numpy().flatten()
            
            # Compute errors
            error = np.abs(c_pred - c_analytical)
            l2_error = np.sqrt(np.mean(error**2))
            linf_error = np.max(error)
            
            res_errors_l2.append(l2_error)
            res_errors_linf.append(linf_error)
        
        errors_l2.append(res_errors_l2)
        errors_linf.append(res_errors_linf)
    
    # Plot convergence
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    for i, t_val in enumerate(time_points):
        plt.plot(resolutions, [errors_l2[j][i] for j in range(len(resolutions))], marker='o', label=f't={t_val:.2f}')
    plt.xlabel('Resolution')
    plt.ylabel('L2 Error')
    plt.title('L2 Error Convergence')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i, t_val in enumerate(time_points):
        plt.plot(resolutions, [errors_linf[j][i] for j in range(len(resolutions))], marker='o', label=f't={t_val:.2f}')
    plt.xlabel('Resolution')
    plt.ylabel('L∞ Error')
    plt.title('L∞ Error Convergence')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return errors_l2, errors_linf


def main():
    # Problem setup
    domain = [-20, 20, -20, 20]  # [x_min, x_max, y_min, y_max] in km
    time_range = [0, 10]  # [t_min, t_max] in seconds
    D = 0.1  # Diffusion coefficient in m²/s
    v = [1.0, 0.5]  # Velocity vector in m/s
    
    # Neural network configuration
    layers = [3, 20, 20, 20, 20, 20, 1]  # [input_dim, hidden_layers, output_dim]
    
    # Training parameters
    batch_sizes = {'pde': 5000, 'ic': 1000, 'bc': 1000}
    lambda_weights = {'pde': 1.0, 'ic': 10.0, 'bc': 10.0}
    learning_rate = 0.001
    epochs = 10000
    
    # Create model
    model = PINN(layers, D, v)
    print(f"Model moved to {next(model.parameters()).device}")
    
    # # Visualize the distribution of collocation points
    # plot_collocation_points(domain, time_range, batch_sizes)
    
    # Train model
    history = train_pinn(model, domain, time_range, batch_sizes, learning_rate, epochs, lambda_weights)
    
    # Evaluate and visualize results
    plot_loss_history(history)
    errors = evaluate_model(model, domain, time_range[1])
    
    # Compute convergence metrics
    errors_l2, errors_linf = compute_convergence_metrics(model, domain, time_range)
    
    print("Mean errors at different time points:")
    for i, t_val in enumerate([0.0, time_range[1]/4, time_range[1]/2, 3*time_range[1]/4, time_range[1]]):
        print(f"t = {t_val:.2f}: {errors[i]:.6f}")


if __name__ == "__main__":
    main()
