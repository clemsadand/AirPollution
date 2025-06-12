import numpy as np
import crbe
import pinn
import meshio
from tqdm import tqdm
import time
import torch
import os
import pandas as pd

os.makedirs("experimental_results", exist_ok=True)
# Reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set up your problem once (outside the loop)
domain = pinn.Domain()

domain_size = 20
n_steps = 128
#mesh_size = 64

# PINN Hyperparamters

lambda_weights = {{'pde': 3.2, 'ic': 2.5365, 'bc': 2.5365}
layers = [3] + [64]*4 + [1]
epochs = 10000
lr = 1e-3

# --- Function to track CPU memory ---
def get_cpu_memory():
    return psutil.Process().memory_info().rss / 1e6  # in MB

# sensitivity setup 
D_list = [0.001, 0.01, 0.1, 1.0, 10]
sensitivity_data = []

filename = "experimental_results/df_sensitivity_data.csv"

mesh_sizes = [4, 8, 16, 32, 64, 128]

for mesh_size in mesh_sizes:
	print(f"Training for mesh size {mesh_size} ...")
	
	# Create mesh only once
	mesh_file = crbe.create_mesh(mesh_size, domain_size=domain_size)
	mesh = meshio.read(mesh_file)
	mesh_data = crbe.MeshData(mesh, domain, nt=n_steps)
	n_ic = round(0.2 * mesh_data.number_of_segments)
	n_bc = n_ic
	n_col = mesh_data.number_of_segments - n_ic - n_bc
	batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_ic}
	
	for i, D in enumerate(D_list):
		print(f"Running for D = {D}")
		#PINN's setup
		pproblem = pinn.Problem(D=D)
		model = pinn.PINN(layers, pproblem, domain, activation="sine").to(device)
		model.train(batch_sizes, epochs, lr, lambda_weights, early_stopping_patience=50, early_stopping_min_delta=1e-6)
		pinn_rel_l2_error, pinn_l2_error, pinn_max_error = model.compute_errors(mesh_data, pproblem.analytical_solution)
		
		print()
		
		#CR-BE setup
		cproblem = crbe.Problem(D=D)
		solver = crbe.BESCRFEM(domain, cproblem, mesh_data, crbe.ElementCR(), 1)
		solver.solve()

		crbe_rel_l2_error, crbe_l2_error, crbe_max_error = solver.compute_errors(cproblem.analytical_solution)
		
		sensitivity_data.append({
		    "mesh_size": mesh_size,
		    "diffusion_coef": D,
		    "pinn_l2_error": pinn_rel_l2_error,
		    "max_error": pinn_max_error,
		    "cr_l2_error": crbe_rel_l2_error,
		    "cr_max_error": crbe_max_error,
		})
		print()
		print("="*50)

df_sensitivity_data = pd.DataFrame(sensitivity_data)

df_sensitivity_data.to_csv(filename)

print(f"Sensitivity analysis ended and results are saved at {filename}")
