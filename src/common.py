import abc
import numpy as np
import torch # Added torch import

# It's good practice to define the backend function here as well,
# as it's used by the Problem class.
def backend(x):
    if isinstance(x, np.ndarray):
        return np
    elif isinstance(x, torch.Tensor): # Corrected from torch.tensor to torch.Tensor
        return torch
    else:
        raise TypeError("Unsupported type")

class AdDifProblem(abc.ABC):
    def __init__(self, v, D):
        self.v = v
        self.D = D

    @abc.abstractmethod
    def initial_condition_fn(self, xyt): # Changed xyt to xy to match pinn.py, but Problem uses xyt for analytical_solution
        pass

    @abc.abstractmethod
    def boundary_fn(self, xyt):
        pass

    @abc.abstractmethod
    def source_term(self, xyt):
        pass

class Problem(AdDifProblem):
    """Physical model definitions and analytical solution."""

    def __init__(self, v=[1.0, 0.5], D=0.1, sigma=1.0):
        super().__init__(v, D)
        """Initialize model parameters."""
        self.sigma = sigma

    def analytical_solution(self, xyt):
        """Compute analytical solution at space-time points."""
        xp = backend(xyt)
        # Ensure xyt has 3 columns (x, y, t) for the operations below
        if xyt.shape[1] != 3:
            raise ValueError("Input xyt must have 3 columns for x, y, and t.")

        denom = 4 * self.D * xyt[:, 2] + self.sigma**2

        num = (xyt[:, 0] - self.v[0] * xyt[:, 2])**2 + (xyt[:, 1] - self.v[1] * xyt[:, 2])**2
        return xp.exp(- num /denom) / (xp.pi * denom)

    def initial_condition_fn(self, xy): # Changed from xyt to xy as per AdDifProblem, but this will need careful checking
        """Evaluate initial condition."""
        xp = backend(xy)
        # This function expects xy (2D space) and adds t=0
        if xy.shape[1] != 2:
            raise ValueError("Input xy for initial_condition_fn must have 2 columns for x and y.")

        if xp == np:
            t = xp.zeros((xy.shape[0], 1), dtype=xp.float32)
            xyt_initial = xp.hstack([xy, t])
        else:
            # Assuming torch tensor for xy
            t = xp.zeros((xy.shape[0], 1), dtype=torch.float32, device=xy.device) # Used torch.float32
            xyt_initial = xp.cat([xy, t], dim=1)

        return self.analytical_solution(xyt_initial)

    def boundary_fn(self, xyt):
        # This function expects xyt (space-time)
        if xyt.shape[1] != 3:
            raise ValueError("Input xyt for boundary_fn must have 3 columns for x, y, and t.")
        return self.analytical_solution(xyt)

    def source_term(self, xyt):
        # This function expects xyt (space-time)
        if xyt.shape[1] != 3:
            raise ValueError("Input xyt for source_term must have 3 columns for x, y, and t.")
        xp = backend(xyt)
        return xp.zeros_like(xyt[:,0])

class Domain:
    """Parameters defining the domain of the problem."""

    def __init__(self, Lx=20, Ly=20, T=10): # Matched defaults from pinn.py
        """Initialize domain parameters."""
        self.Lx = Lx
        self.Ly = Ly
        self.T = T

    def is_boundary(self, x): # x here is expected to be spatial points, typically (N, 2)
        """Check if points are on boundary."""
        # This function expects x (2D or 3D space)
        # For 2D spatial boundary checks, x should have 2 columns.
        if x.shape[1] < 2:
            raise ValueError("Input x for is_boundary must have at least 2 columns for x and y.")

        is_left = np.isclose(x[:, 0], -self.Lx, atol=1e-10)
        is_right = np.isclose(x[:, 0], self.Lx, atol=1e-10)
        is_bottom = np.isclose(x[:, 1], -self.Ly, atol=1e-10)
        is_top = np.isclose(x[:, 1], self.Ly, atol=1e-10)
        # If x has a time component, it's ignored here.
        return is_left | is_right | is_bottom | is_top
