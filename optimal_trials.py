import numpy as np
import torch
import time
import pinn
import crbe
import meshio

# Reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

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

epochs = 500  # Fewer epochs for faster search

best_trial = {'depth': 6, 'width': 64, 'lr': 0.001996124072912953, 'lambda_pde': 4.700967312082998, 'lambda_ic_bc': 1.4612500492945448, 'activation': 'swish'}

depth = best_trial["depth"]
width = best_trial["width"]
lr = best_trial["lr"]
lambda_pde = best_trial['lambda_pde']
lambda_ic_bc = best_trial['lambda_ic_bc']
activation = best_trial['activation']

# Build model
layers = [3] + [width] * depth + [1]
lambda_weights = {'pde': lambda_pde, 'ic': lambda_ic_bc, 'bc': lambda_ic_bc}
model = pinn.PINN(layers, problem, domain, activation=activation)

start_time = time.time()
model.train(batch_sizes, epochs, lr, lambda_weights,
                    early_stopping_patience=10,
                    early_stopping_min_delta=1e-5)
rel_l2_eror, l2_error, max_error = model.compute_errors(mesh_data, problem.analytical_solution)
train_time = time.time() - start_time


print(f"Compute error\n\tRel L2 Error: {rel_l2_eror:.4f}\n\tL2 Error: {l2_error:.4f}\n\tMax Error: {max_error:.4f}")
print()

model.plot_interpolated_solution(10.0, mesh_data, problem.analytical_solution)
