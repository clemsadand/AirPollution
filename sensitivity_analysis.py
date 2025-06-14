import numpy as np
import crbe
import pinn
import meshio
from tqdm import tqdm
import time
import torch
import os
import pandas as pd
import argparse

# Reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

# --- Parse Command Line Arguments ---
parser = argparse.ArgumentParser(description="PINN experiment with configurable network.")
parser.add_argument('--width', type=int, default=4, help='Number of hidden layers in the neural network')
#parser.add_argument('--depth', type=int, default=64, help='Number of neurons per layers in the neural network')
parser.add_argument('--activation', type=str, default="tanh", help='Type of activation (tanh, sine, swish)')
parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs')
parser.add_argument('--early_stopping_patience', type=int, default=50000, help='Number of epochs to wait if no improvement')
parser.add_argument('--restore_best_weights', type=bool, default=True, help='Wether to restore best model or not')
#parser.add_argument('--learning_rate', type=float, default=3e-3, help='Learning rate')
#-------------------------------------
args = parser.parse_args()
width = args.width
#depth = args.depth
activation = args.activation
early_stopping_patience = args.early_stopping_patience
epochs = args.epochs
restore_best_weights = args.restore_best_weights
#learning_rate = args.learning_rate
# ------------------------------------

base_dir = f"experimental_results_sensibility_analysis"

# Check if the directory exists
if os.path.exists(base_dir):
    # Append current date and time to create a new unique folder
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"{base_dir}_{date_str}"
else:
    exp_dir = base_dir

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set up your problem once (outside the loop)
domain = pinn.Domain()

domain_size = 20
n_steps = 128
#mesh_size = 64

# PINN Hyperparamters

lambda_weights = {'pde': 180.0, 'ic': 80.0, 'bc': 80.0}
#layers = [3] + [depth] * width + [1]


# --- Function to track CPU memory ---
def get_cpu_memory():
    return psutil.Process().memory_info().rss / 1e6  # in MB

# sensitivity setup 
D_list = [0.001, 0.01, 0.1, 1.0, 10]
sensitivity_data = []

filename = f"{exp_dir}/df_sensitivity_data.csv"

mesh_sizes = [4, 8, 16, 32, 64, 128]
n_neurons = [2, 4, 8, 16, 32, 64]
lr_list = [3e-4, 3e-4, 2e-4, 4e-5, 1e-4, 1e-4]
epochs_list = [1000, 2000, 4000, 8000, 16000, 32000]

for j, mesh_size in enumerate(mesh_sizes):
	print(f"Training for mesh size {mesh_size} ...")

	#PINN hyperparmas
	layers = [3] + [n_neurons[j]] * width + [1]
	lr = lr_list[j]
	early_stopping_patience = epochs_list[j]
	
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
		model = pinn.PINN(layers, pproblem, domain, activation=activation).to(device)
		model.train(batch_sizes, epochs, lr, lambda_weights, early_stopping_patience=early_stopping_patience, early_stopping_min_delta=1e-6, restore_best_weights=restore_best_weights)
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
		#break
	#break

df_sensitivity_data = pd.DataFrame(sensitivity_data)

df_sensitivity_data.to_csv(filename)

print(f"Sensitivity analysis ended and results are saved at {filename}")
