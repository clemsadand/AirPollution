# --- Imports ---
import numpy as np
import crbe 
import pinn 
import meshio
from tqdm import tqdm
import time
import torch
import psutil
import pandas as pd
import os
import gc
from datetime import datetime
import argparse

# Reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

# --- Parse Command Line Arguments ---
parser = argparse.ArgumentParser(description="PINN experiment with configurable network width.")
parser.add_argument('--width', type=int, default=4, help='Number of hidden layers in the neural network')
parser.add_argument('--activation', type=str, default="tanh", help='Type of activation (tanh, sine, swish)')
parser.add_argument('--restore_best_weights', type=bool, default=True, help='Wether to restore best model or not')
parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs')
#parser.add_argument('--early_stopping_patience', type=int, default=1000, help='Number of epochs')
#---------------------------------------
args = parser.parse_args()
width = args.width
activation = args.activation
restore_best_weights = args.restore_best_weights
epochs = args.epochs
#early_stopping_patience = args.early_stopping_patience
#---------------------------------------
base_dir = f"pinn_experimental_results"

# Check if the directory exists
if os.path.exists(base_dir):
    # Append current date and time to create a new unique folder
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"{base_dir}_{date_str}"
else:
    exp_dir = base_dir

# Create the directory
os.makedirs(exp_dir, exist_ok=True)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- Function to track GPU and CPU memory ---
def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6  # in MB
    return 0

def get_cpu_memory():
    return psutil.Process().memory_info().rss / 1e6  # in MB

# --- Problem Setup ---
domain = pinn.Domain()
problem = pinn.Problem(sigma=1.0)

# --- Experimental Settings ---
domain_size = 20
lambda_weights = {'pde': 180.0, 'ic': 80.0, 'bc': 80.0}
n_steps = 128

mesh_sizes = [4, 8, 16, 32, 64, 128]
n_neurons = [2, 4, 8, 16, 32, 64]

epochs_list = [500, 1000, 2000, 4000, 8000, 16000]
early_stopping_patience_list = [500, 500, 500, 1000, 1000, 1000]
lr_list = [3e-4, 3e-4, 2e-4, 4e-5, 1e-4, 1e-4]

# logging
n_dofs = []
n_boundary_dofs = []
pinn_results = []
result_history = {}

# --- Main Loop ---
for i in range(len(mesh_sizes)):

    # Hyperparameters' setup
    #layers = [3] + layers_list[i] + [1]
    layers = [3] + [n_neurons[i]] * width + [1]
    epochs = epochs_list[i]
    early_stopping_patience = early_stopping_patience_list[i]
    learning_rate = lr_list[i]
    
    #generate mesh
    mesh_size = mesh_sizes[i]
    mesh_file = crbe.create_mesh(mesh_size, domain_size=domain_size)
    mesh = meshio.read(mesh_file)
    mesh_data = crbe.MeshData(mesh, domain, nt=n_steps)

    #store 
    n_dofs.append(mesh_data.number_of_segments)
    n_boundary_dofs.append(len(mesh_data.boundary_segments))

    #define batch size
    n_col = round(mesh_data.number_of_segments / 1.4)
    n_ic = round(0.2 * n_col)
    n_bc = round(0.2 * n_col)
    batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_ic}

    #Define model
    model = pinn.PINN(layers, problem, domain, activation=activation).to(device)
    
    print(f"Training for mesh size {mesh_size} ...")

    start_time = time.time()

    # Reset memory tracking
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    initial_cpu_memory = get_cpu_memory()
    initial_gpu_memory = get_gpu_memory()

    history = model.train(batch_sizes, epochs, learning_rate, lambda_weights, early_stopping_patience=early_stopping_patience)
    
    train_time = time.time() - start_time

    final_gpu_memory = get_gpu_memory()
    final_cpu_memory = get_cpu_memory()

    result_history[f"mesh_size_{mesh_size}"] = history

    rel_l2_error, l2_error, max_error = model.compute_errors(mesh_data, problem.analytical_solution)
    
    model.plot_interpolated_solution(10.0, mesh_data, analytical_sol_fn=problem.analytical_solution, save_dir=exp_dir, name=f"ms{mesh_size}_pinn")
    
    model.plot_history(save_dir=exp_dir, name=f"ms{mesh_size}_pinn")
		
    pinn_results.append({
        "mesh_size": mesh_size,
        "n_dofs": mesh_data.number_of_segments,
        "n_boundary_dofs": len(mesh_data.boundary_segments),
        "rel_l2_error": rel_l2_error,
        "l2_error": l2_error,
        "max_error": max_error,
        "train_time": train_time,
        "final_loss": history["total_loss"][-1],
        "number_of_collocation_points": mesh_data.number_of_segments,
        "n_parameters": sum(l1 * l2 + l2 for l1, l2 in zip(layers[:-1], layers[1:])),
        "gpu_memory_usage_MB": final_gpu_memory - initial_gpu_memory,
        "cpu_memory_usage_MB": final_cpu_memory - initial_cpu_memory,
    })

    print(f"Mesh size: {mesh_size}")
    print(f"GPU Memory: {final_gpu_memory - initial_gpu_memory:.2f} MB")
    print(f"CPU Memory: {final_cpu_memory - initial_cpu_memory:.2f} MB")
    print("-" * 40)

    del model
    
    if mesh_size >=32:
        pd.DataFrame(pinn_results).to_csv(f"{exp_dir}/df_pinn_training_results.csv")
    #break
    #if mesh_size==32:
    #	break

# --- Export Results ---
df_pinn = pd.DataFrame(pinn_results)

df_pinn.to_csv(f"{exp_dir}/df_pinn_training_results.csv")
print(df_pinn)















