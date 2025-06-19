# Air Pollution Modelling with PINN and CRBE

## Description

This project study the modeling of air pollution using the advection-diffusion equations in a bounded two-dimensional domain. It implements and compares two numerical methods for solving time-dependent advection-diffusion equations:
1.  **Physics-Informed Neural Networks (PINN)**: A deep learning approach where a neural network is trained to satisfy the PDE, initial conditions, and boundary conditions.
2.  **Crouzeix-Raviart Finite Element Method (CRBE)**: A classical finite element approach using Crouzeix-Raviart elements and a Backward Euler time-stepping scheme.

The project includes scripts for running experiments to compare the accuracy and performance of these methods.

## Features

-   Implementation of CRBE for 2D advection-diffusion problems.
-   Implementation of PINNs for 2D advection-diffusion problems.
-   Mesh generation for FEM using the Gmsh Python API.
-   Latin Hypercube Sampling for generating collocation points in PINNs.
-   Experiment scripts (`crbe_experiments.py`, `pinn_experiments.py`) for systematic comparison.
-   Solution visualization and error analysis (L2 error, max error).
-   Early stopping and learning rate scheduling for PINN training.
-   Support for different activation functions in PINN (Tanh, Sine, Swish).

## Methodologies

### Physics-Informed Neural Networks (PINN)

PINNs leverage deep neural networks to approximate the solution of PDEs. The network takes spatio-temporal coordinates (x, y, t) as input and outputs the concentration c(x,y,t). The training process minimizes a composite loss function that includes:
1.  The PDE residual over a set of collocation points in the domain.
2.  The error in satisfying initial conditions.
3.  The error in satisfying boundary conditions.

The `pinn.py` script defines the PINN architecture, training loop, and evaluation methods.

Key components:
-   `Problem`: Defines the analytical solution, initial, and boundary conditions (similar to CRBE but using PyTorch tensors).
-   `Domain`: Specifies the physical domain parameters.
-   `PINN`: Defines the neural network architecture, methods for computing the PDE residual, and the training loop.
-   `EarlyStopping`: Utility to prevent overfitting.
-   Helper functions for sampling collocation points (LHS) and computing gradients.


### Crouzeix-Raviart fem and Bacwkard Euler scheme (CRBE)

The CRBE approach discretizes the spatial domain using a triangular mesh. The concentration variable is approximated by piecewise linear functions that are continuous at the midpoints of element edges (Crouzeix-Raviart elements). A Backward Euler scheme is used for time discretization. The `crbe.py` script handles mesh creation, matrix assembly, solution, and error computation.

Key components:
-   `create_mesh()`: Generates a square mesh using `gmsh`.
-   `Problem`: Defines the analytical solution, initial conditions, boundary conditions, and source term.
-   `Domain`: Specifies the physical domain parameters.
-   `MeshData`: Stores and processes mesh information.
-   `ElementCR`: Defines properties of the Crouzeix-Raviart reference element.
-   `BESCRFEM`: Implements the Backward Euler Scheme with CRBE, including matrix assembly, solving the linear system, and computing errors.

## File Structure

-   `./`: Contains the core solver implementations.
    -   `pinn.py`: Implementation of the Physics-Informed Neural Network solver.
    -   `crbe.py`: Implementation of the Crouzeix-Raviart Finite Element Method solver.
-   `utils/`: Contains the auxiliary implementations.
    -   `utils/common.py`: Contains common class definitions like `Problem`, `Domain`, and `AdDifProblem`, used by both `pinn.py` and `crbe.py`.
-   `experiments/`: Contains scripts for running various experiments, comparisons, and hyperparameter searches (e.g., `pinn_experiments.py`, `crbe_experiments.py`, `fixed_runtime_experiments.py`, `optimal_hyperparams_search.py`).
-   `scripts/`: Contains utility or specific problem scripts.
    -   `scripts/problem3.py`: Example script demonstrating a specific problem setup.
-   `requirements.txt`: Lists the Python dependencies for the project.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/clemsadand/AirPollution.git
    cd AirPollution
    ```
2.  **Create a virtual environment (optionnal):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y libglu1-mesa # additional packages for gmsh
    pip install -r requirements.txt
    ```
4.  **Gmsh:** This project uses `gmsh` for mesh generation. If it's not installed correctly via pip or if you encounter issues, you might need to install it system-wide. Please refer to the [official Gmsh website](https://gmsh.info/) for installation instructions.

## Usage

### Running Individual Solvers

Both `crbe.py` and `pinn.py` can be run directly to solve a default problem and generate solution plots:

-   **CRBE Solver:**
    ```bash
    python crbe.py
    ```
    This will generate a mesh, solve the problem, print error metrics, and save solution plots in the `results/` directory.

-   **PINN Solver:**
    ```bash
    python pinn.py
    ```
    This will train the PINN, print error metrics (evaluated on a mesh generated by `crbe.create_mesh`), plot the training history, and save solution plots in the `results/` directory. Note that PINN training can be computationally intensive.

### Running Experiments

The experiment scripts automate running the solvers with different configurations and log the results:
-   **PINN Experiments:**
    ```bash
    python -m experiments.pinn_experiments --width=4 --epochs=0 --activation=tanh --restore_best_weights=True 
    ```
    This script will train and evaluate PINNs with different network architectures (number of neurons) and evaluation mesh sizes. Note that setting `epochs=0` will use default value defined for each mesh size. Results will be saved to `pinn_experimental_results/`.
-   **CRBE Experiments:**
    ```bash
    python -m experiments.crbe_experiments
    ```
    This script will run the CRBE solver for various mesh sizes, and the results (errors, timings, memory usage) will be saved to `crbe_experimental_results/`.
-   **Sensitivity Analysis:**
    ```bash
    python -m experiments.sensitivity_analysis --width=4 --epochs=0 --activation=tanh
    ```
    This script will run the CRBE solver for various mesh sizes, and the results (errors, timings, memory usage) will be saved to `experimental_results_sensibility_analysis/`.
-   **Fixed Runtime Analysis:**
    ```bash
    python -m experiments.fixed_runtime_experiments
    ```
    This script will run the CRBE solver for various mesh sizes, and the results (errors, timings, memory usage) will be saved to `experimental_results_sensibility_analysis/`.


