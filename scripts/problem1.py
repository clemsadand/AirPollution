import torch
import numpy as np
from src import crbe # Changed import
from src.common import AdDifProblem # Changed import
from src import pinn # Changed import
import meshio
from tqdm import tqdm
import time
import psutil
import pandas as pd
torch.manual_seed(1234)
np.random.seed(1234)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


#*******************************
def backend(x):
    if isinstance(x, np.ndarray):
        return np
    elif isinstance(x, torch.Tensor):
        return torch
    else:
        raise TypeError("Unsupported type")

class Problem1(AdDifProblem):
    def __init__(self, v=[1.0, 0.5], D=0.1):
        super().__init__(v, D)

    def initial_condition_fn(self, xy):
        xp = backend(xy)
        return xp.exp(-5 * xp.sqrt((xy[:,0] + 10.0)**2 + (xy[:,1] + 10.0)**2))
    
    def boundary_fn(self, xyt):
        xp = backend(xyt)
        return xp.zeros_like(xyt[:,0])
    
    def source_term(self, xyt):
        xp = backend(xyt)
        return xp.zeros_like(xyt[:,0])
        
#***************************************
problem1 = Problem1() #define an advection-diffusion problem
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
"""
cr_element = crbe.ElementCR() 

solver1 = crbe.BESCRFEM(domain, problem1, mesh_data, cr_element, 1)

solutions1 = solver1.solve()

solver1.plot_interpolated_solution(name="crbe1")
# solver1.plot_solution()
"""
#************************************************
n_col = round(mesh_data.number_of_segments / 1.4)
n_ic = round(0.2 * n_col)
n_bc = round(0.2 * n_col)
batch_sizes = {'pde': n_col, 'ic': n_ic, 'bc': n_ic}


lambda_weights = {'pde': 20.0, 'ic': 10.0, 'bc': 5.00}

lr = 1e-3
epochs = 1000
layers = [3] + [16] * 4 + [1]

#define pinn's model
model = pinn.PINN(layers, problem1, domain).to(device)

model.train(
    batch_sizes, 
    epochs, 
    lr, 
    lambda_weights, 
    early_stopping_patience=10, 
    early_stopping_min_delta=1e-6,
    restore_best_weights=True
)

model.plot_history(name="pinn1")
_ = model.plot_interpolated_solution(10.0, mesh_data, name="pinn1")
