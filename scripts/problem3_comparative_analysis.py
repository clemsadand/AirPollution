# Standard library imports
import torch
import numpy as np
import time
import psutil
import pandas as pd
import os
import gc
import argparse
import shutil

# Project-specific imports
from scripts.problem3 import Problem
import crbe
import pinn
import meshio

# Set seeds for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_cpu_memory():
    """Returns current CPU RSS memory usage in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

def get_gpu_memory():
    """Returns peak GPU memory allocated on the 'device' in MB since last reset."""
    return torch.cuda.max_memory_allocated(device) / 1e6

def main():
    print("Starting comparative analysis for Problem 3...")
    
    mesh_sizes = [4, 8, 16, 32, 64, 128]
    n_neurons = [2, 4, 8, 16, 32, 64]
    epochs_list = [400, 800, 1600, 3200, 6400, 12000]# if not epochs else [epochs]*len(mesh_sizes)
    #early_stopping_patience_list = [500, 500, 1000, 1000, 1000]
    lr_list = [1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4]
    lambda_weights = {'pde': 1.0, 'ic': 8.0, 'bc': 1.0}
    lr = 1e-3
    epochs = 10 # Using 3000 epochs as per plan
    
    results_data = []
    
    exp_dir = "problem3_analysis_results"
    os.makedirs(exp_dir, exist_ok=True)

    # Common problem definition (outside the loop)
    problem = Problem()
    domain = crbe.Domain() 
    
    # Domain size and number of time steps 
    d_size = 20  # domain size
    n_steps = 128 # number of time steps

    for i, m_size in enumerate(mesh_sizes):
        print(f"\n--- Processing Mesh Size: {m_size} ---")
        current_run_data = {'m_size': m_size}

        # --- Mesh Setup ---
        mesh_file = crbe.create_mesh(m_size, domain_size=d_size)
        mesh = meshio.read(mesh_file)#generat mesh data
        mesh_data = crbe.MeshData(mesh, domain, nt=n_steps)
				
        print(f"Mesh data processed for m_size={m_size}.")

        # --- CRBE Execution ---
        print(f"\n--- Running CRBE for m_size={m_size} ---")
        
        cr_element = crbe.ElementCR()
        crbe_solver = crbe.BESCRFEM(domain, problem, mesh_data, cr_element, 1)
        
        gc.collect()
        if torch.cuda.is_available():
        	torch.cuda.empty_cache()
        	torch.cuda.reset_peak_memory_stats(device)
        initial_crbe_cpu_mem = get_cpu_memory()
        
        start_time = time.time()
        all_crbe_solutions = crbe_solver.solve()
        crbe_time = time.time() - start_time
        
        final_crbe_cpu_mem = get_cpu_memory()
        peak_crbe_gpu_mem = get_gpu_memory()
        crbe_cpu_mem_used = final_crbe_cpu_mem - initial_crbe_cpu_mem
        
        u_crbe_final_midpoints = all_crbe_solutions[-1, :].copy()
        current_run_data.update({
            'crbe_time_solve_s': crbe_time, 'crbe_cpu_mem_diff_MB': crbe_cpu_mem_used,
            'crbe_gpu_mem_peak_MB': peak_crbe_gpu_mem, 'crbe_status': 'success'
        })
        print(f"CRBE solve (m_size={m_size}) finished: Time {crbe_time:.2f}s, CPU Mem {crbe_cpu_mem_used:.2f}MB, GPU Mem {peak_crbe_gpu_mem:.2f}MB")

        del crbe_solver, all_crbe_solutions, cr_element 
        gc.collect()

        # --- PINN Execution ---
        print(f"\n--- Running PINN for m_size={m_size} ---")
        
        layers = [3] + [n_neurons[i]] * 3 + [1]
        n_col_float = mesh_data.number_of_segments / 1.4
        n_col = int(round(n_col_float))
        n_ic = int(round(0.25 * n_col))
        n_bc = int(round(0.15 * n_col))

        batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_bc}
        
        epochs = epochs_list[i]
        lr = lr_list[i]
        pinn_model = pinn.PINN(layers, problem, domain).to(device)
        
        gc.collect()
        if torch.cuda.is_available():
        	torch.cuda.empty_cache()
        	torch.cuda.reset_peak_memory_stats(device)
        initial_pinn_cpu_mem = get_cpu_memory()
        
        start_time = time.time()
        history = pinn_model.train(
				batch_sizes, 
				epochs, 
				lr, 
				lambda_weights, 
				early_stopping_patience=0, 
				early_stopping_min_delta=1e-6,
				restore_best_weights=True
		)
        pinn_time = time.time() - start_time
        
        final_pinn_cpu_mem = get_cpu_memory()
        peak_pinn_gpu_mem = get_gpu_memory()
        pinn_cpu_mem_used = final_pinn_cpu_mem - initial_pinn_cpu_mem
        epochs_run = len(history['pde_loss'])
        
        midpoints_tensor = torch.tensor(mesh_data.midpoints, dtype=torch.float32, device=device)
        t_final_tensor = torch.full((midpoints_tensor.shape[0], 1), domain.T, dtype=torch.float32, device=device)
        xyt_final = torch.cat([midpoints_tensor, t_final_tensor], dim=1)
        u_pinn_final_midpoints_tensor = pinn_model(xyt_final).squeeze()
        u_pinn_final_midpoints = u_pinn_final_midpoints_tensor.detach().cpu().numpy()

        current_run_data.update({
            'pinn_time_train_s': pinn_time, 'pinn_cpu_mem_diff_MB': pinn_cpu_mem_used,
            'pinn_gpu_mem_peak_MB': peak_pinn_gpu_mem, 'pinn_epochs_run': epochs_run,
            'pinn_status': 'success'
        })
        print(f"PINN training (m_size={m_size}) finished: Time {pinn_time:.2f}s ({epochs_run} epochs), CPU Mem {pinn_cpu_mem_used:.2f}MB, GPU Mem {peak_pinn_gpu_mem:.2f}MB")
        del pinn_model, history, midpoints_tensor, t_final_tensor, xyt_final, u_pinn_final_midpoints_tensor
        gc.collect()

        # --- Error Calculation ---
        error_diff_abs = np.abs(u_pinn_final_midpoints - u_crbe_final_midpoints)
        l2_error_diff = np.linalg.norm(error_diff_abs)
        max_error_diff = np.max(error_diff_abs)
        current_run_data.update({'l2_error_diff': l2_error_diff, 'max_error_diff': max_error_diff, 'error_status': 'success'})
        print(f"Error (m_size={m_size}): L2 Diff = {l2_error_diff:.4e}, Max Diff = {max_error_diff:.4e}")
        
        results_data.append(current_run_data)
        
        if m_size == 64:
        	break

    # --- Save Results ---
    df_results = pd.DataFrame(results_data)
    results_filename = os.path.join(exp_dir, "problem3_comparative_analysis_by_mesh_size.csv")
    df_results.to_csv(results_filename, index=False)
    print(f"\nResults saved to {results_filename}")
    print(df_results)

    print("\nComparative analysis script finished.")

if __name__ == "__main__":
    main()
