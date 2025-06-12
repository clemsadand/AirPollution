import matplotlib.pyplot as plt
import numpy as np
import crbe
import meshio
from tqdm import tqdm
import time
import psutil
import torch
import pandas as pd
import gc
import torch
import os

torch.manual_seed(1234)
np.random.seed(1234)

os.makedirs("experimental_results", exist_ok=True)


# --- Problem Setup ---
domain_size = 20.0

domain = crbe.Domain()
problem = crbe.Problem()

mesh_sizes = [4, 8, 16, 32, 64, 128]
n_steps = 128
crbe_results = []
cr_element = crbe.ElementCR()

# --- Function to track CPU memory ---
def get_cpu_memory():
    return psutil.Process().memory_info().rss / 1e6  # in MB

# --- Loop over mesh sizes ---
for i, mesh_size in enumerate(mesh_sizes):
    # --- Prepare for memory tracking ---
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # initial_gpu_memory = get_gpu_memory()
    initial_cpu_memory = get_cpu_memory()

    print(f"Training for mesh size = {mesh_size} ...")
    start_time = time.time()

    # --- Mesh and Solver Setup ---
    mesh_file = crbe.create_mesh(mesh_size, domain_size=domain_size)
    mesh = meshio.read(mesh_file)
    mesh_data = crbe.MeshData(mesh, domain, nt=n_steps)

    
    solver = crbe.BESCRFEM(domain, problem, mesh_data, cr_element, time_scheme_order=1) 

    solver.solve()
    train_time = time.time() - start_time

    # --- Memory tracking after solve ---
    gc.collect()
    final_cpu_memory = get_cpu_memory()
    # final_gpu_memory = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0

    rel_l2_error, l2_error, max_error = solver.compute_errors(problem.analytical_solution)
    solver.plot_interpolated_solution(analytical_sol_fn=problem.analytical_solution, save_dir="experimental_results", name=f"ms{mesh_size}_crbe")
    
    # --- Save results ---
    crbe_results.append({
        "mesh_size": mesh_size,
        "n_dofs": mesh_data.number_of_segments,
        "n_boundary_dofs": len(mesh_data.boundary_segments),
        "l2_error": l2_error,
        "rel_l2_error": rel_l2_error,
        "max_error": max_error,
        "mesh_size": mesh_size,
        "train_time": train_time,
        "gpu_memory_usage_MB": 0,
        "cpu_memory_usage_MB": final_cpu_memory - initial_cpu_memory,
        "number_of_collocation_points": mesh_data.number_of_segments,
    })

    # --- Print summary ---
    print(f"Mesh size: {mesh_size}")
    # print(f"Peak GPU Memory: {final_gpu_memory:.2f} MB")
    print(f"CPU Memory Used: {final_cpu_memory - initial_cpu_memory:.2f} MB")
    print("-" * 40)
    # break

# --- Results as DataFrame ---
df_crbe = pd.DataFrame(crbe_results)
df_crbe.to_csv("experimental_results/df_crbe_training_results.csv")
df_crbe
