# --- Fixed Runtime Comparison Experiment ---
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
import matplotlib.pyplot as plt
import argparse

# Reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

# --- Parse Command Line Arguments ---
parser = argparse.ArgumentParser(description="PINN experiment with configurable network width.")
parser.add_argument('--run_for_testing', type=bool, default=False, help='Number of hidden layers in the neural network')
args = parser.parse_args()
run_for_testing = args.run_for_testing
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

save_dir = "experimental_results/fixed_runtime"
os.makedirs(save_dir, exist_ok=True)

# --- Memory tracking functions ---
def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6  # in MB
    return 0

def get_cpu_memory():
    return psutil.Process().memory_info().rss / 1e6  # in MB

# --- Problem Setup ---
domain_pinn = pinn.Domain()
problem_pinn = pinn.Problem(sigma=1.0)
domain_crbe = crbe.Domain()
problem_crbe = crbe.Problem(sigma=1.0)

# --- Experimental Settings ---
domain_size = 20
n_steps = 128
mesh_sizes = [4, 8, 16, 32, 64]  # Reduced for faster testing
time_budgets = [30, 60, 120, 180] if not run_for_testing else [10] # Time budgets in seconds

# PINN settings
lambda_weights = {'pde': 180.0, 'ic': 80.0, 'bc': 80.0}
#learning_rate = 3e-3
lr_list = [3e-4, 3e-4, 2e-4, 4e-5, 1e-4, 1e-4]
base_neurons = [2, 4, 8, 16, 32] # Corresponding to mesh sizes

# CRBE settings
cr_element = crbe.ElementCR()

def run_pinn_with_time_budget(mesh_data, time_budget, n_neurons, lr):
    """Run PINN training for a fixed time budget"""
    layers = [3] + [n_neurons] * 4 + [1]
    
    # Define batch sizes
    n_col = round(mesh_data.number_of_segments / 1.4)
    n_ic = round(0.2 * n_col)
    n_bc = round(0.2 * n_col)
    batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_ic}
    
    model = pinn.PINN(layers, problem_pinn, domain_pinn).to(device)
    
    # Track memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    initial_cpu_memory = get_cpu_memory()
    initial_gpu_memory = get_gpu_memory()
    
    start_time = time.time()
    epoch = 0
    history = {"total_loss": [], "pde_loss": [], "ic_loss": [], "bc_loss": []}
    
    print(f"PINN training with {time_budget}s budget...")
    
    while (time.time() - start_time) < time_budget:
        # Single epoch training
        epoch_history = model.train(batch_sizes, epochs=1, lr=lr, lambda_weights=lambda_weights)
        
        # Store history
        for key in history.keys():
            history[key].extend(epoch_history[key])
        
        epoch += 1
        
        # Print progress every 100 epochs
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch}, Elapsed: {elapsed:.1f}s, Loss: {history['total_loss'][-1]:.6f}")
    
    actual_runtime = time.time() - start_time
    final_gpu_memory = get_gpu_memory()
    final_cpu_memory = get_cpu_memory()
    
    # Compute errors
    rel_l2_error, l2_error, max_error = model.compute_errors(
        mesh_data, problem_pinn.analytical_solution)
    
    result = {
        "method": "PINN",
        "actual_runtime": actual_runtime,
        "epochs_completed": epoch,
        "final_loss": history["total_loss"][-1] if history["total_loss"] else float('inf'),
        "rel_l2_error": rel_l2_error,
        "l2_error": l2_error,
        "max_error": max_error,
        "n_parameters": sum(l1 * l2 + l2 for l1, l2 in zip(layers[:-1], layers[1:])),
        "gpu_memory_usage_MB": final_gpu_memory - initial_gpu_memory,
        "cpu_memory_usage_MB": final_cpu_memory - initial_cpu_memory,
        "convergence_history": history["total_loss"]
    }
    
    del model
    return result

def run_crbe_with_time_budget(mesh_data, time_budget):
    """Run CRBE solver (should complete within time budget)"""
    
    # Track memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    initial_cpu_memory = get_cpu_memory()
    
    print(f"CRBE solving...")
    start_time = time.time()
    
    solver = crbe.BESCRFEM(domain_crbe, problem_crbe, mesh_data, cr_element, time_scheme_order=1)
    solver.solve()
    
    actual_runtime = time.time() - start_time
    
    # Check if within time budget
    if actual_runtime > time_budget:
        print(f"  Warning: CRBE took {actual_runtime:.1f}s, exceeding budget of {time_budget}s")
    
    rel_l2_error, l2_error, max_error = solver.compute_errors(problem_crbe.analytical_solution)
    
    gc.collect()
    final_cpu_memory = get_cpu_memory()
    
    result = {
        "method": "CRBE",
        "actual_runtime": actual_runtime,
        "epochs_completed": 1,  # CRBE solves directly
        "final_loss": None,  # CRBE doesn't have loss function
        "rel_l2_error": rel_l2_error,
        "l2_error": l2_error,
        "max_error": max_error,
        "n_parameters": mesh_data.number_of_segments,  # Approximate parameter count
        "gpu_memory_usage_MB": 0,
        "cpu_memory_usage_MB": final_cpu_memory - initial_cpu_memory,
        "convergence_history": None
    }
    
    return result

# --- Main Experiment Loop ---
all_results = []

for mesh_idx, mesh_size in enumerate(mesh_sizes):
    print(f"\n{'='*50}")
    print(f"MESH SIZE: {mesh_size}")
    print(f"{'='*50}")
    
    # Create mesh
    mesh_file = crbe.create_mesh(mesh_size, domain_size=domain_size)
    mesh = meshio.read(mesh_file)
    mesh_data = crbe.MeshData(mesh, domain_crbe, nt=n_steps)
    
    n_neurons = base_neurons[mesh_idx]
    lr = lr_list[mesh_idx]
    
    for time_budget in time_budgets:
        print(f"\nTime Budget: {time_budget}s")
        print("-" * 30)
        
        # Run PINN
        pinn_result = run_pinn_with_time_budget(mesh_data, time_budget, n_neurons, lr)
        pinn_result.update({
            "mesh_size": mesh_size,
            "time_budget": time_budget,
            "n_dofs": mesh_data.number_of_segments,
            "n_boundary_dofs": len(mesh_data.boundary_segments),
        })
        all_results.append(pinn_result)
        
        # Run CRBE
        crbe_result = run_crbe_with_time_budget(mesh_data, time_budget)
        crbe_result.update({
            "mesh_size": mesh_size,
            "time_budget": time_budget,
            "n_dofs": mesh_data.number_of_segments,
            "n_boundary_dofs": len(mesh_data.boundary_segments),
        })
        all_results.append(crbe_result)
        
        print(f"PINN  - Runtime: {pinn_result['actual_runtime']:.1f}s, "
              f"Epochs: {pinn_result['epochs_completed']}, "
              f"Rel L2 Error: {pinn_result['rel_l2_error']:.6f}")
        print(f"CRBE  - Runtime: {crbe_result['actual_runtime']:.1f}s, "
              f"Rel L2 Error: {crbe_result['rel_l2_error']:.6f}")

# --- Save Results ---
df_results = pd.DataFrame(all_results)
df_results.to_csv(f"{save_dir}/fixed_runtime_comparison.csv", index=False)

print(f"\n{'='*50}")
print("EXPERIMENT COMPLETED")
print(f"{'='*50}")
print(f"Results saved to: {save_dir}/fixed_runtime_comparison.csv")
print(f"Total experiments: {len(all_results)}")

# --- Display Summary ---
print("\nSUMMARY:")
summary_stats = df_results.groupby(['method', 'time_budget']).agg({
    'rel_l2_error': ['mean', 'std'],
    'actual_runtime': ['mean', 'std'],
    'epochs_completed': 'mean'
}).round(6)

summary_stats.to_csv(f"{save_dir}/fixed_runtime_summary_stats.csv")
print(summary_stats)

print("\nExperiment completed successfully!")
