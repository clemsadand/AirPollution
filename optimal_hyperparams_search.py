import optuna
import numpy as np
import torch
import time
import pinn
import crbe
import meshio
import os

# Reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

# Set up your problem once (outside the loop)
domain = pinn.Domain()
problem = pinn.Problem()
domain_size = 20
n_steps = 128
mesh_sizes = [64, 128]


epochs = 200

def objective(trial):
    # Sample hyperparameters
    depth = trial.suggest_int('depth', 3, 6)
    width = trial.suggest_categorical('width', [16, 32, 64])
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    lambda_pde = trial.suggest_float('lambda_pde', 0.1, 10.0, log=True)
    lambda_ic_bc = trial.suggest_float('lambda_ic_bc', 0.1, 10.0, log=True)
    activation = trial.suggest_categorical('activation', ['tanh', 'sine', 'swish'])

    # Build model
    layers = [3] + [width] * depth + [1]
    lambda_weights = {'pde': lambda_pde, 'ic': lambda_ic_bc, 'bc': lambda_ic_bc}
    model = pinn.PINN(layers, problem, domain, activation=activation)

    try:
        start_time = time.time()
        model.train(batch_sizes, epochs, lr, lambda_weights,
                    early_stopping_patience=10,
                    early_stopping_min_delta=1e-5)
        rel_l2_error, l2_error, max_error = model.compute_errors(mesh_data, problem.analytical_solution)
        train_time = time.time() - start_time
        trial.set_user_attr("train_time", train_time)
        return l2_error + max_error  # Objective to minimize
    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")  # Penalize failures

for mesh_size in mesh_sizes:
# Create mesh only once
mesh_file = crbe.create_mesh(mesh_size, domain_size=domain_size)
mesh = meshio.read(mesh_file)
mesh_data = crbe.MeshData(mesh, domain, nt=n_steps)

n_ic = round(0.2 * mesh_data.number_of_segments)
n_bc = n_ic
n_col = mesh_data.number_of_segments - n_ic - n_bc
batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_ic}


study = optuna.create_study(direction="minimize", study_name="pinn-hpo")
#study.optimize(objective, n_trials=100)  # You can increase this later
study.optimize(objective, n_trials=100, n_jobs=os.cpu_count())

# Save results
import pandas as pd
df_results = study.trials_dataframe()
df_results.to_csv("optuna_pinn_results_{mesh_size}.csv", index=False)

# Show best hyperparameters
print("Best trial:")
print(study.best_trial.params)
