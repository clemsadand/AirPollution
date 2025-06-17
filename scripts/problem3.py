import torch
import numpy as np
import crbe
from crbe import AdDifProblem
import pinn
import meshio
from tqdm import tqdm
import time
import psutil
import pandas as pd
import os

torch.manual_seed(1234)
np.random.seed(1234)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



def backend(x):
    if isinstance(x, np.ndarray):
        return np
    elif isinstance(x, torch.Tensor):
        return torch
    else:
        raise TypeError("Unsupported type")

class Problem(AdDifProblem):
    def __init__(self, v=[1.0, 0.0], D=0.1):
        super().__init__(v, D)

    def initial_condition_fn(self, xy):
        xp = backend(xy)
        cond_x = (xy[:, 0] >= 8.0) & (xy[:, 0] <= 12.0)
        cond_y = (xy[:, 1] >= 8.0) & (xy[:, 1] <= 12.0)
        return xp.where(cond_x & cond_y, xp.ones_like(xy[:, 0]), xp.zeros_like(xy[:, 0]))
    
    def boundary_fn(self, xyt):
        xp = backend(xyt)
        return xp.zeros_like(xyt[:, 0])
    
    def source_term(self, xyt):
        xp = backend(xyt)
        return xp.zeros_like(xyt[:, 0])

if __name__ == "__main__":
				  
		#***************************************
		problem = Problem() #define an advection-diffusion problem
		domain = crbe.Domain()

		#create a mesh
		d_size = 20 #domain size
		m_size = 64 # mesh size
		n_steps = 128 # number of time steps

		mesh_file = crbe.create_mesh(m_size, domain_size=d_size)
		mesh = meshio.read(mesh_file)
		#generat mesh data
		mesh_data = crbe.MeshData(mesh, domain, nt=n_steps)

		#************************************************

		#Type of finite element methods

		cr_element = crbe.ElementCR() 

		solver1 = crbe.BESCRFEM(domain, problem, mesh_data, cr_element, 1)

		solutions1 = solver1.solve()


		# solver1.plot_solution()

		#************************************************
		n_col = round(mesh_data.number_of_segments / 1.4)
		n_ic = round(0.35 * n_col)
		n_bc = round(0.05 * n_col)
		batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_bc}


		lambda_weights = {'pde': 1, 'ic': 8.0, 'bc': 1.0}#(10, 100, 20); (20, 90, 10)

		lr = 1e-3
		epochs = 3000
		layers = [3] + [30] * 3  + [1] # depth = 3, width = 28 ou 30

		#define pinn's model
		model = pinn.PINN(layers, problem, domain).to(device)

		model.train(
				batch_sizes, 
				epochs, 
				lr, 
				lambda_weights, 
				early_stopping_patience=10, 
				early_stopping_min_delta=1e-6,
				restore_best_weights=True
		)

		model.plot_history(name="pinn3")

		time_indices = [0, 64, n_steps-1]
		for it in time_indices:
				solver1.plot_interpolated_solution(time_index=it,name="crbe3")
				t = mesh_data.time_discr[it]
				_ = model.plot_interpolated_solution(t, mesh_data, name="pinn3")


		midpoints = torch.tensor(mesh_data.midpoints, dtype=torch.float32, device=device)
		t_tensor = torch.full((midpoints.shape[0], 1), mesh_data.domain.T, dtype=torch.float32, device=device)
		xyt = torch.cat([midpoints, t_tensor], dim=1)
				        
		with torch.no_grad():
			u_pinn_midpoints = model(xyt).squeeze()

		u_pinn_midpoints = u_pinn_midpoints.detach().cpu().numpy()
		u_crbe_midpoints = solver1.solutions[-1, :]

		error = np.abs(u_pinn_midpoints - u_crbe_midpoints)
		# l2_error = np.sqrt(np.sum(error**2))
		# max_error = max(error)

		l2_error = np.linalg.norm(error) #/ np.linalg.norm(u_crbe)
		max_error = np.max(np.abs(error))

		print()
		print("L2 error: ",l2_error)
		print("Max error: ",max_error)

