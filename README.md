# Air Pollution Modeling: Comparative Analysis of Numerical Methods

This project studies the modeling of air pollution using the advection-diffusion equation in a bounded two-dimensional domain. We implement and compare two numerical methods for solving time-dependent advection-diffusion equations:

- **Physics-Informed Neural Networks (PINN)**: A deep learning approach where a neural network is trained to satisfy the PDE, initial conditions, and boundary conditions
- **Crouzeix-Raviart Finite Element Method (CRBE)**: A classical finite element approach using Crouzeix-Raviart elements and a Backward Euler time-stepping scheme

## Mathematical Problem

The project solves the 2D advection-diffusion equation:

$$
\partial_t c + \mathbf v \cdot \nabla c - D \Delta c = s(x, y, t) \quad \text{in } 
$$

where:
- $c(x,y,t)$ is the concentration field
- $\mathbf v$ is the velocity field
- $D$ is the diffusion coefficient
- $s(x,y,t)$ is the source term

The equation is solved subject to appropriate initial and boundary conditions in a bounded 2D domain.

## Features

- **Dual Implementation Approach**:
  - CRBE implementation for 2D advection-diffusion problems
  - PINN implementation for 2D advection-diffusion problems
- **Advanced Meshing**: Mesh generation for FEM using the Gmsh Python API
- **Smart Sampling**: Latin Hypercube Sampling for generating collocation points in PINNs
- **Systematic Experimentation**: Automated experiment scripts (`crbe_experiments.py`, `pinn_experiments.py`) for comprehensive method comparison
- **Comprehensive Analysis**: Solution visualization, L2 error analysis, and maximum error computation
- **Optimized Training**: Early stopping and learning rate scheduling for PINN training
- **Flexible Architecture**: Support for different activation functions in PINN (Tanh, Sine, Swish)
- **Performance Monitoring**: Memory usage and timing analysis for both methods

## Repository Structure

```
AirPollution/
├── crbe.py                     # Crouzeix-Raviart Finite Element implementation
├── pinn.py                     # Physics-Informed Neural Network implementation
├── experiments/
│   ├── crbe_experiments.py     # CRBE systematic experiments
│   ├── pinn_experiments.py     # PINN systematic experiments
│   ├── sensitivity_analysis.py # Parameter sensitivity analysis
│   └── fixed_runtime_experiments.py # Fixed runtime comparison
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── experimental_results/       # Generated files with experiment results
└── results/                    # Generated plots from individual solver runs
```

## Method Overview

### Physics-Informed Neural Networks (PINNs)

PINNs leverage deep neural networks to approximate the solution of PDEs. The network takes spatio-temporal coordinates $(x, y, t)$ as input and outputs the concentration $c(x,y,t)$. The training process minimizes a composite loss function that includes:

- **PDE Residual**: Evaluated over a set of collocation points in the domain
- **Initial Condition Error**: Ensures proper initial state satisfaction
- **Boundary Condition Error**: Enforces boundary constraints

**Key Components**:
- `Problem`: Defines the initial, boundary conditions, and source term using Numpy arrays and/or PyTorch tensors
- `Domain`: Specifies the physical domain parameters
- `PINN`: Defines the neural network architecture, PDE residual computation, and training loop
- `EarlyStopping`: Utility to prevent overfitting
- Helper functions for Latin Hypercube Sampling and automatic differentiation

### Crouzeix-Raviart Finite Element Method (CRBE)

The CRBE approach discretizes the spatial domain using a uniform triangular mesh. The concentration variable is approximated by piecewise linear functions that are continuous at the midpoints of element edges (Crouzeix-Raviart elements). A Backward Euler scheme provides temporal discretization.

**Key Components**:
- `create_mesh()`: Generates structured triangular meshes using `gmsh`
- `Problem`: Defines the initial, boundary conditions, and source term using Numpy arrays and/or PyTorch tensors
- `Domain`: Specifies physical domain parameters
- `MeshData`: Stores and processes mesh information
- `ElementCR`: Defines Crouzeix-Raviart reference element properties
- `BESCRFEM`: Implements the complete Backward Euler Scheme with matrix assembly and error computation

## Requirements

- **Python Version**: 3.8+
- **Key Dependencies**:
  - `torch>=1.9.0` (PyTorch for PINN implementation)
  - `numpy>=1.21.0`
  - `matplotlib>=3.4.0`
  - `scipy>=1.7.0`
  - `gmsh>=4.8.0` (mesh generation)
  - `pyDOE` (for Latin Hypercube Sampling)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/clemsadand/AirPollution.git
   cd AirPollution
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install system dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y libglu1-mesa  # Additional packages for gmsh
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

**Note on Gmsh**: This project uses `gmsh` for mesh generation. If you encounter installation issues via pip, please refer to the [official Gmsh website](https://gmsh.info/) for system-specific installation instructions.

## Usage

### Quick Start

Both solvers can be run directly to solve a default problem and generate solution plots:

#### CRBE Solver
```bash
python crbe.py
```
This generates a mesh, solves the advection-diffusion problem, prints error metrics, and saves solution plots in the `results/` directory.

#### PINN Solver
```bash
python pinn.py
```
This trains the PINN, evaluates it on a validation mesh, plots training history, and saves solution plots in the `results/` directory. Note that PINN training can be computationally intensive.

### Systematic Experiments

The experiment scripts automate running the solvers with different configurations:

#### PINN Experiments
```bash
python -m experiments.pinn_experiments --width=4 --epochs=0 --activation=tanh --restore_best_weights=True
```
- Trains PINNs with different network architectures and evaluation mesh sizes
- Setting `epochs=0` uses default values optimized for each mesh size
- Results saved to `experimental_results/pinn/`

#### CRBE Experiments
```bash
python -m experiments.crbe_experiments
```
- Runs CRBE solver for various mesh sizes
- Results saved to `experimental_results/crbe/`

#### Sensitivity Analysis
```bash
python -m experiments.sensitivity_analysis --width=4 --epochs=0 --activation=tanh
```
- Analyzes parameter sensitivity for both methods
- Results saved to `experimental_results/sensitivity/`

#### Fixed Runtime Analysis
```bash
python -m experiments.fixed_runtime_experiments
```
- Compares methods under fixed computational budgets
- Results saved to `experimental_results/fixed_runtime/`

## Sample Results

The project validates both methods against analytical solutions. Typical performance characteristics:

- **CRBE**: O(h²) convergence rate in space, robust for various mesh sizes
- **PINN**: Highly dependent on network architecture and training parameters, can achieve spectral accuracy with sufficient training

Error metrics include L2 norm and maximum absolute error computed over the entire spatio-temporal domain.

## Key Parameters

### PINN Configuration
- **Network Architecture**: Fully connected layers with configurable width
- **Activation Functions**: Tanh (default), Sine, Swish
- **Training**: Adam optimizer with learning rate scheduling
- **Collocation Points**: Latin Hypercube Sampling for optimal coverage

### CRBE Configuration
- **Element Type**: Crouzeix-Raviart (non-conforming linear elements)
- **Time Stepping**: Backward Euler (first-order implicit) or Cranck-Nicolson (second-order implicit)
- **Mesh Generation**: Structured triangular meshes via Gmsh

## License

This project is available under the MIT License.

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
- Crouzeix, M., & Raviart, P. A. (1973). Conforming and nonconforming finite element methods for solving the stationary Stokes equations I. *Revue française d'automatique, informatique, recherche opérationnelle*, 7(3), 33-75.
