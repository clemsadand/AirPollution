# crbe_fixed_time_experiments.py
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
import os

torch.manual_seed(1234)
np.random.seed(1234)

os.makedirs("experimental_results", exist_ok=True)

# --- Problem Setup ---
domain_size = 20.0
domain = crbe.Domain()
problem = crbe.Problem()
exact_sol_fn = problem.analytical_solution

n_steps = 128
cr_element = crbe.ElementCR()

# Time budget in seconds
fixed_time_budget = 60

# --- Function to track CPU memory ---
def get_cpu_memory():
    return psutil.Process().memory_info().rss / 1e6  # in MB

# --- Determine the largest mesh size within time budget ---
crbe_results = []
mesh_sizes = [4, 8, 16, 32, 64, 128]

for mesh_size in mesh_sizes:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    initial_cpu_memory = get_cpu_memory()
    print(f"Training CRBE for mesh size = {mesh_size} ...")

    start_time = time.time()
    mesh_file = crbe.create_mesh(mesh_size, domain_size=domain_size)
    mesh = meshio.read(mesh_file)
    mesh_data = crbe.MeshData(mesh, domain, nt=n_steps)

    solver = crbe.BESCRFEM(domain, problem, mesh_data, cr_element, time_scheme_order=1)
    solver.solve()
    elapsed_time = time.time() - start_time

    if elapsed_time > fixed_time_budget:
        print(f"Exceeded time budget ({elapsed_time:.2f}s > {fixed_time_budget}s). Skipping.")
        break

    rel_l2_error, l2_error, max_error, _, _ = solver.compute_errors(exact_sol_fn)
    final_cpu_memory = get_cpu_memory()

    crbe_results.append({
        "mesh_size": mesh_size,
        "n_dofs": mesh_data.number_of_segments,
        "n_boundary_dofs": len(mesh_data.boundary_segments),
        "l2_error": l2_error,
        "rel_l2_error": rel_l2_error,
        "max_error": max_error,
        "train_time": elapsed_time,
        "gpu_memory_usage_MB": 0,
        "cpu_memory_usage_MB": final_cpu_memory - initial_cpu_memory,
        "number_of_collocation_points": mesh_data.number_of_segments,
    })

    print(f"Mesh size: {mesh_size} completed in {elapsed_time:.2f} s")
    print("-" * 40)

# Save results
pd.DataFrame(crbe_results).to_csv("experimental_results/crbe_fixed_time_results.csv")

