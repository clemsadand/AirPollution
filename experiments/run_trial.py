import numpy as np
import torch
import time
from src import pinn # Changed import
from src import crbe # Changed import
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

n_col = round(mesh_data.number_of_segments / 1.4)
n_ic = round(0.2 * n_col)
n_bc = round(0.2 * n_col)
batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_ic}


layers = [3] + [16] * 5 + [1]  # Input: (x, y, t) → Output: c(x, y, t)
model = pinn.PINN(layers, problem, domain).to(device)

params = {'lr': 0.0008361816272135304, 'lambda_pde': 0.48669353902173246, 'lambda_ic_bc': 0.2029249101415861}
lambda_weights = {'pde': params["lambda_pde"], 'ic': params["lambda_ic_bc"], 'bc': params["lambda_ic_bc"]}
    
lr = params["lr"]
epochs = 4000
early_stopping_patience = 10000

model.train(batch_sizes, epochs, lr, lambda_weights, early_stopping_patience=early_stopping_patience, early_stopping_min_delta=1e-6, restore_best_weights=True)

model.plot_history()
    
errors = model.compute_errors(mesh_data, problem.analytical_solution)
print(f"Compute error\n\tRel L2 Error: {errors[0]:.4f}\n\tL2 Error: {errors[1]:.4f}\n\tMax Error: {errors[2]:.4f}")
print()
    
model.plot_interpolated_solution(10.0, mesh_data, problem.analytical_solution)
