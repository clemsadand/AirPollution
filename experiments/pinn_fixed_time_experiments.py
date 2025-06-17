# pinn_fixed_time_experiments.py
import numpy as np
from src import crbe # Changed import
from src import pinn # Changed import
import meshio
from tqdm import tqdm
import time
import torch
import psutil
import pandas as pd
import os

torch.manual_seed(1234)
np.random.seed(1234)

os.makedirs("experimental_results", exist_ok=True)

# --- Problem Setup ---
domain = pinn.Domain()
problem = pinn.Problem()
domain_size = 20

lambda_weights = {'pde': 1.0, 'ic': 10.0, 'bc': 10.0}
learning_rate = 1e-3
n_steps = 128
mesh_size = 64  # Fixed mesh size for consistency
n_neurons = 32   # Moderate network size

fixed_time_budget = 60  # in seconds

# --- Function to track GPU and CPU memory ---
def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6  # in MB
    return 0

def get_cpu_memory():
    return psutil.Process().memory_info().rss / 1e6  # in MB

# --- Prepare experiment ---
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

initial_cpu_memory = get_cpu_memory()
initial_gpu_memory = get_gpu_memory()

layers = [3] + [n_neurons] * 4 + [1]
mesh_file = crbe.create_mesh(mesh_size, domain_size=domain_size)
mesh = meshio.read(mesh_file)
mesh_data = crbe.MeshData(mesh, domain, nt=n_steps)

n_col = round(mesh_data.number_of_segments / 1.4)
n_ic = round(0.2 * n_col)
n_bc = round(0.2 * n_col)
batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_ic}

model = pinn.PINN(layers, problem, domain)

print(f"Training PINN for fixed time budget of {fixed_time_budget}s ...")

start_time = time.time()
train_time = 0
history = {"total_loss": []}

# --- Custom timed training loop ---
while train_time < fixed_time_budget:
    epoch_start = time.time()
    stop_training = model.train_one_epoch(batch_sizes, learning_rate, lambda_weights)
    epoch_time = time.time() - epoch_start
    train_time = time.time() - start_time
    history["total_loss"].append(model.loss_history[-1])
    if stop_training:
        break

final_gpu_memory = get_gpu_memory()
final_cpu_memory = get_cpu_memory()

rel_l2_error, l2_error, max_error, u_num, u_exact = model.compute_errors(mesh_data, problem.analytical_solution)

results = pd.DataFrame([{
    "mesh_size": mesh_size,
    "n_dofs": mesh_data.number_of_segments,
    "n_boundary_dofs": len(mesh_data.boundary_segments),
    "rel_l2_error": rel_l2_error,
    "l2_error": l2_error,
    "max_error": max_error,
    "train_time": train_time,
    "final_loss": history["total_loss"][-1] if history["total_loss"] else None,
    "n_parameters": sum(l1 * l2 + l2 for l1, l2 in zip(layers[:-1], layers[1:])),
    "gpu_memory_usage_MB": final_gpu_memory - initial_gpu_memory,
    "cpu_memory_usage_MB": final_cpu_memory - initial_cpu_memory,
    "number_of_collocation_points": mesh_data.number_of_segments,
}])

results.to_csv("experimental_results/pinn_fixed_time_results.csv")
print("Training complete and results saved.")

