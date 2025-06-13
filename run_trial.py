import numpy as np
import torch
import time
import pinn
import crbe
import meshio

# Reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up your problem once (outside the loop)
domain = pinn.Domain()
problem = pinn.Problem()
domain_size = 20
n_steps = 128
mesh_size = 64

# Create mesh only once
mesh_file = crbe.create_mesh(mesh_size, domain_size=domain_size)
mesh = meshio.read(mesh_file)
mesh_data = crbe.MeshData(mesh, domain, nt=n_steps)

n_ic = round(0.2 * mesh_data.number_of_segments)
n_bc = n_ic
n_col = mesh_data.number_of_segments - n_ic - n_bc
batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_ic}


layers = [3] + [16] * 4 + [1]  # Input: (x, y, t) → Output: c(x, y, t)
model = pinn.PINN(layers, problem, domain).to(device)

# Compute residual
n_ic = round(0.2 * mesh_data.number_of_segments)
n_bc = n_ic
n_col = mesh_data.number_of_segments - n_ic - n_bc
batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_ic}
lambda_weights = {'pde': 1.4990615182382665, 'ic': 0.49929070971734624, 'bc': 0.49929070971734624}
    
lr = 0.008257554820761981
epochs = 4000
    
model.train(batch_sizes, epochs, lr, lambda_weights, early_stopping_patience=100, early_stopping_min_delta=1e-6, restore_best_weights=False)

model.plot_history()
    
errors = model.compute_errors(mesh_data, problem.analytical_solution)
print(f"Compute error\n\tRel L2 Error: {errors[0]:.4f}\n\tL2 Error: {errors[1]:.4f}\n\tMax Error: {errors[2]:.4f}")
print()
    
model.plot_interpolated_solution(10.0, mesh_data, problem.analytical_solution)
