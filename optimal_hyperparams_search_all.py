import optuna
import numpy as np
import torch
import time
import pinn
import crbe
import meshio
import os
import time

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
epochs = 2000
#*****************
activation = 'tanh'
depth = 4



def objective(trial):
    # Sample hyperparameters (same for all widths)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    lambda_pde = trial.suggest_float('lambda_pde', 0.1, 10.0, log=True)
    lambda_ic_bc = trial.suggest_float('lambda_ic_bc', 0.1, 10.0, log=True)

    lambda_weights = {'pde': lambda_pde, 'ic': lambda_ic_bc, 'bc': lambda_ic_bc}
    widths = [8, 16, 32, 64, 128]

    errors = []
    for width in widths:
        try:
            layers = [3] + [width] * depth + [1]
            model = pinn.PINN(layers, problem, domain, activation=activation).to(device)

            model.train(batch_sizes, epochs, lr, lambda_weights,
                        early_stopping_patience=100,
                        early_stopping_min_delta=1e-7,
                        restore_best_weights=True)

            rel_l2_error, l2_error, max_error = model.compute_errors(mesh_data, problem.analytical_solution)
            error = (l2_error - 1e-5)**2 + (max_error - 1e-5)**2
            errors.append(error)

        except Exception as e:
            print(f"Trial failed for width {width}: {e}")
            errors.append(float("inf"))

    # Return average error over all widths
    mean_error = np.mean(errors)
    trial.set_user_attr("mean_error", mean_error)
    return mean_error


start_ = time.time()
study = optuna.create_study(direction="minimize", study_name="pinn-hpo")
#study.optimize(objective, n_trials=100)  # You can increase this later
study.optimize(objective, n_trials=20, n_jobs=os.cpu_count())
end_ = time.time()


print(f"\nMinization ended in {end_ - start_:0.2f}")

# Save results
import pandas as pd
df_results = study.trials_dataframe()
df_results.to_csv("optuna_pinn_results.csv", index=False)

# Show best hyperparameters
print("Best trial:")
print(study.best_trial.params)
