import optuna
import numpy as np
import torch
import time
import pinn
import crbe
import meshio
import os
import time

import argparse

#***********************************************************
width = 16
n_trials=10
epochs = 10000

parser = argparse.ArgumentParser(description="PINN experiment.")
parser.add_argument('--width', type=int, default=width, help='Neural network width')
parser.add_argument('--n_trials', type=int, default=n_trials, help='Number of trials')
parser.add_argument('--epochs', type=int, default=epochs, help='Number of epochs')

args = parser.parse_args()
width = args.width
n_trials= args.n_trials
epochs = args.epochs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

#*****************
activation = 'tanh'
depth = 4



def objective(trial):
    # Sample hyperparameterss
    #depth = trial.suggest_int('depth', 3, 6)
    #width = trial.suggest_categorical('width', [8, 16, 32, 64, 128])
    lr = trial.suggest_float('lr', 1e-4, 5e-1, log=True)
    lambda_pde = trial.suggest_float('lambda_pde', 0.1, 10.0, log=True)
    lambda_ic_bc = trial.suggest_float('lambda_ic_bc', 0.1, 10.0, log=True)
    #activation = trial.suggest_categorical('activation', ['tanh', 'sine', 'swish'])

    # Build model
    layers = [3] + [width] * depth + [1]
    lambda_weights = {'pde': lambda_pde, 'ic': lambda_ic_bc, 'bc': lambda_ic_bc}
    model = pinn.PINN(layers, problem, domain, activation=activation).to(device)

    try:
        start_time = time.time()
        model.train(batch_sizes, epochs, lr, lambda_weights,
                    early_stopping_patience=1000,
                    early_stopping_min_delta=1e-7,
                    restore_best_weights=True)
        rel_l2_error, l2_error, max_error = model.compute_errors(mesh_data, problem.analytical_solution)
        train_time = time.time() - start_time
        trial.set_user_attr("train_time", train_time)
        return (l2_error - 1e-5)**2 + (max_error - 1e-5)**2  # Objective to minimize
    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")  # Penalize failures

start_ = time.time()
study = optuna.create_study(direction="minimize", study_name="pinn-hpo")
#study.optimize(objective, n_trials=100)  # You can increase this later
study.optimize(objective, n_trials=n_trials, n_jobs=os.cpu_count())
end_ = time.time()


print(f"\nMinization ended in {end_ - start_:0.2f}")

# Save results
import pandas as pd
df_results = study.trials_dataframe()
df_results.to_csv(f"optuna_pinn_results_{width}.csv", index=False)

# Show best hyperparameters
print("Best trial:")
print(study.best_trial.params)
