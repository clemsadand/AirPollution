import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time

# Vérification de la disponibilité du GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilisation du dispositif: {device}")

# Définition de la classe de réseau neuronal pour PINNs
class PINNModel(nn.Module):
    def __init__(self, layers, activation=nn.Tanh()):
        super(PINNModel, self).__init__()

        # Construction des couches du réseau
        self.activation = activation
        self.loss_function = nn.MSELoss(reduction='mean')

        # Couches du réseau
        self.layers = nn.ModuleList()

        # Couche d'entrée
        self.layers.append(nn.Linear(3, layers[0]))

        # Couches cachées
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        # Couche de sortie
        self.layers.append(nn.Linear(layers[-1], 1))

        # Initialisation des poids
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass à travers le réseau
        x: entrée de forme [batch_size, 3] où 3 représente (t, x, y)
        """
        for i in range(len(self.layers)-1):
            x = self.activation(self.layers[i](x))

        # Couche de sortie (sans activation)
        x = self.layers[-1](x)

        return x

class PINN:
    def __init__(self, layers, lb, ub, nu=0.01, beta=[1.0, 0.5], lr=0.001):
        """
        Initialisation du réseau de neurones pour PINNs

        Args:
            layers: liste des tailles des couches cachées du réseau
            lb: bornes inférieures [t_min, x_min, y_min]
            ub: bornes supérieures [t_max, x_max, y_max]
            nu: coefficient de diffusion
            beta: vecteur de vitesse d'advection [beta_x, beta_y]
        """
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        self.nu = nu
        self.beta = beta

        # Initialisation du modèle
        self.model = PINNModel(layers).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=100, verbose=True)

    def net_u(self, t, x, y):
        """Prédit la solution u(t,x,y)"""
        # Concaténation des entrées
        X = torch.cat([t, x, y], dim=1)

        # Normalisation des entrées entre -1 et 1
        X_norm = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

        # Prédiction du réseau
        u = self.model(X_norm)

        return u

    def net_pde(self, t, x, y):
        """Calcule les résidus de l'équation d'advection-diffusion"""
        t.requires_grad_(True)
        x.requires_grad_(True)
        y.requires_grad_(True)

        # Prédiction de u
        u = self.net_u(t, x, y)

        # Calcul de du/dt
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calcul de du/dx
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calcul de du/dy
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calcul de d²u/dx²
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # Calcul de d²u/dy²
        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]

        # Résidu de l'équation d'advection-diffusion:
        # du/dt + beta_x * du/dx + beta_y * du/dy - nu * (d²u/dx² + d²u/dy²) = 0
        pde = u_t + self.beta[0] * u_x + self.beta[1] * u_y - self.nu * (u_xx + u_yy)

        return pde

    def loss_fn(self, t_data, x_data, y_data, u_data, t_col, x_col, y_col):
        """Calcule la fonction de perte totale"""
        # Prédiction aux points de données
        u_pred = self.net_u(t_data, x_data, y_data)
        # MSE sur les données
        loss_data = torch.mean((u_pred - u_data)**2)

        # Résidus de la PDE aux points de collocation
        pde_residual = self.net_pde(t_col, x_col, y_col)
        # MSE sur les résidus de la PDE
        loss_pde = torch.mean(pde_residual**2)

        # Perte totale = perte sur les données + perte sur la PDE
        loss = loss_data + loss_pde

        return loss, loss_data, loss_pde

    def train_step(self, t_data, x_data, y_data, u_data, t_col, x_col, y_col):
        """Effectue une étape d'entraînement"""
        # Mise à zéro des gradients
        self.optimizer.zero_grad()

        # Calcul de la perte
        loss, loss_data, loss_pde = self.loss_fn(t_data, x_data, y_data, u_data, t_col, x_col, y_col)

        # Rétropropagation
        loss.backward()

        # Mise à jour des poids
        self.optimizer.step()


        return loss.item(), loss_data.item(), loss_pde.item()

    def train(self, t_data, x_data, y_data, u_data, t_col, x_col, y_col, epochs, verbose=True):
        """Entraîne le modèle"""
        start_time = time.time()
        history = {'loss': [], 'loss_data': [], 'loss_pde': []}

        for epoch in range(epochs):
            loss, loss_data, loss_pde = self.train_step(t_data, x_data, y_data, u_data, t_col, x_col, y_col)

            history['loss'].append(loss)
            history['loss_data'].append(loss_data)
            history['loss_pde'].append(loss_pde)

            # Mise à jour du scheduler
            self.scheduler.step(loss)

            if verbose and epoch % 100 == 0:
                elapsed = time.time() - start_time
                print(f'Epoch: {epoch}, Loss: {loss:.6}, Data Loss: {loss_data:.6e}, PDE Loss: {loss_pde:.6e}, Time: {elapsed:.2f} s')
                start_time = time.time()

        return history

    def predict(self, t, x, y):
        """Prédit la solution aux points donnés"""
        self.model.eval()
        with torch.no_grad():
            u = self.net_u(t, x, y)
        return u.cpu().numpy()

# Fonctions pour les solutions exactes
def exact_solution(t, x, y, nu=0.01, beta=[1.0, 0.5]):
    """
    Solution exacte pour le problème de la Gaussienne en mouvement
    u(t,x,y) = exp(-(x-beta_x*t)^2/(4*nu*t) - (y-beta_y*t)^2/(4*nu*t))/(4*pi*nu*t)
    """
    # Éviter division par zéro à t=0
    t = np.maximum(t, 1e-10)

    # Centre mobile
    x_c = x - beta[0] * t
    y_c = y - beta[1] * t

    # Solution analytique
    numerator = np.exp(-(x_c**2 + y_c**2) / (4 * nu * t))
    denominator = 4 * np.pi * nu * t
    return numerator / denominator

def exact_solution_gaussian_pulse(t, x, y, nu=0.01, beta=[1.0, 0.5], sigma=0.1):
    """
    Solution exacte pour une impulsion gaussienne sous advection-diffusion
    u(t,x,y) = exp(-(x-beta_x*t)^2/(4*nu*t+sigma^2) - (y-beta_y*t)^2/(4*nu*t+sigma^2))/(4*pi*(nu*t+sigma^2/4))
    """
    # Centre mobile
    x_c = x - beta[0] * t
    y_c = y - beta[1] * t

    # Variance qui augmente avec le temps
    sigma_t2 = 4 * nu * t + sigma**2

    # Solution analytique
    numerator = np.exp(-(x_c**2 + y_c**2) / sigma_t2)
    denominator = np.pi * sigma_t2
    return numerator / denominator



def generate_training_data(t_domain, x_domain, y_domain, exact_sol_fn, n_ic=20, n_bc=20, n_col=1000):
    """
    Génère les données d'entraînement pour le problème d'advection-diffusion

    Args:
        t_domain: [t_min, t_max]
        x_domain: [x_min, x_max]
        y_domain: [y_min, y_max]
        exact_sol_fn: fonction de solution exacte
        n_ic: nombre de points pour la condition initiale
        n_bc: nombre de points pour les conditions aux limites
        n_col: nombre de points de collocation

    Returns:
        t_data, x_data, y_data, u_data: tenseurs PyTorch pour les conditions aux limites
        t_col, x_col, y_col: tenseurs PyTorch pour les points de collocation
    """
    # Domaine
    t_min, t_max = t_domain
    x_min, x_max = x_domain
    y_min, y_max = y_domain

    # Points de collocation (intérieur du domaine)
    t_col = np.random.uniform(t_min, t_max, n_col).reshape(-1, 1)
    x_col = np.random.uniform(x_min, x_max, n_col).reshape(-1, 1)
    y_col = np.random.uniform(y_min, y_max, n_col).reshape(-1, 1)

    # Points pour la condition initiale (t=0)
    t_ic = np.zeros((n_ic**2, 1))
    x_ic = np.linspace(x_min, x_max, n_ic).reshape(-1, 1)
    y_ic = np.linspace(y_min, y_max, n_ic).reshape(-1, 1)
    x_ic, y_ic = np.meshgrid(x_ic, y_ic)
    x_ic = x_ic.flatten().reshape(-1, 1)
    y_ic = y_ic.flatten().reshape(-1, 1)
    u_ic = exact_sol_fn(t_ic, x_ic, y_ic).reshape(-1, 1)

    # Points pour les conditions aux limites (frontières du domaine spatial)
    # Limite x=x_min
    t_bc1 = np.random.uniform(t_min, t_max, n_bc).reshape(-1, 1)
    x_bc1 = np.ones((n_bc, 1)) * x_min
    y_bc1 = np.random.uniform(y_min, y_max, n_bc).reshape(-1, 1)
    u_bc1 = exact_sol_fn(t_bc1, x_bc1, y_bc1).reshape(-1, 1)

    # Limite x=x_max
    t_bc2 = np.random.uniform(t_min, t_max, n_bc).reshape(-1, 1)
    x_bc2 = np.ones((n_bc, 1)) * x_max
    y_bc2 = np.random.uniform(y_min, y_max, n_bc).reshape(-1, 1)
    u_bc2 = exact_sol_fn(t_bc2, x_bc2, y_bc2).reshape(-1, 1)

    # Limite y=y_min
    t_bc3 = np.random.uniform(t_min, t_max, n_bc).reshape(-1, 1)
    x_bc3 = np.random.uniform(x_min, x_max, n_bc).reshape(-1, 1)
    y_bc3 = np.ones((n_bc, 1)) * y_min
    u_bc3 = exact_sol_fn(t_bc3, x_bc3, y_bc3).reshape(-1, 1)

    # Limite y=y_max
    t_bc4 = np.random.uniform(t_min, t_max, n_bc).reshape(-1, 1)
    x_bc4 = np.random.uniform(x_min, x_max, n_bc).reshape(-1, 1)
    y_bc4 = np.ones((n_bc, 1)) * y_max
    u_bc4 = exact_sol_fn(t_bc4, x_bc4, y_bc4).reshape(-1, 1)

    # Concaténation des données
    t_data = np.vstack([t_ic, t_bc1, t_bc2, t_bc3, t_bc4])
    x_data = np.vstack([x_ic, x_bc1, x_bc2, x_bc3, x_bc4])
    y_data = np.vstack([y_ic, y_bc1, y_bc2, y_bc3, y_bc4])
    u_data = np.vstack([u_ic, u_bc1, u_bc2, u_bc3, u_bc4])

    # Conversion en tenseurs PyTorch
    t_data = torch.tensor(t_data, dtype=torch.float32).to(device)
    x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_data = torch.tensor(y_data, dtype=torch.float32).to(device)
    u_data = torch.tensor(u_data, dtype=torch.float32).to(device)

    t_col = torch.tensor(t_col, dtype=torch.float32).to(device)
    x_col = torch.tensor(x_col, dtype=torch.float32).to(device)
    y_col = torch.tensor(y_col, dtype=torch.float32).to(device)

    return t_data, x_data, y_data, u_data, t_col, x_col, y_col

def compute_error(model, t, x, y, exact_sol_fn):
    """Calcule l'erreur entre la prédiction du modèle et la solution exacte"""
    # Conversion en tenseurs PyTorch
    t_tensor = torch.tensor(t, dtype=torch.float32).to(device)
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    # Prédiction du modèle
    u_pred = model.predict(t_tensor, x_tensor, y_tensor)

    # Solution exacte
    u_exact = exact_sol_fn(t, x, y).reshape(-1, 1)

    # Calcul des erreurs
    error = np.abs(u_pred - u_exact)
    rel_l2_error = np.sqrt(np.mean(np.square(error))) / (np.sqrt(np.mean(np.square(u_exact))) + 1e-10)
    max_error = np.max(error)

    return rel_l2_error, max_error, u_pred, u_exact

def plot_solution(t_val, x_eval, y_eval, u_pred, u_exact, title):
    """
    Plot side-by-side contour plots of the numerical (PINN) solution and the exact solution
    without showing the error.
    
    Args:
        t_val: Time value
        x_eval: Grid points in x direction
        y_eval: Grid points in y direction
        u_pred: Predicted solution from PINN
        u_exact: Exact analytical solution
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    # Reshape solutions for plotting
    X, Y = np.meshgrid(x_eval, y_eval)
    U_pred = u_pred.reshape(X.shape)
    U_exact = u_exact.reshape(X.shape)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Determine common color scale for both plots
    vmin = min(np.min(U_pred), np.min(U_exact))
    vmax = max(np.max(U_pred), np.max(U_exact))
    
    # Plot numerical solution (PINN)
    c1 = ax1.contourf(X, Y, U_pred, 50, cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Numerical Solution (PINN) at t = {t_val}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    # ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot exact solution
    c2 = ax2.contourf(X, Y, U_exact, 50, cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Exact Solution at t = {t_val}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    # ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(c2, cax=cbar_ax, label='u(t,x,y)')
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    return fig
  
# Fonction principale
def main():
    print("Démarrage de l'implémentation PINNs pour l'équation d'advection-diffusion 2D avec PyTorch")

    # Paramètres du problème
    nu = 0.1  # coefficient de diffusion
    beta = [1.0, 0.5]  # vitesse d'advection [beta_x, beta_y]

    # Domaines
    T = 10.0
    t_domain = [0.0, T]
    x_domain = [-20.0, 20.0]
    y_domain = [-20.0, 20.0]

    # Bornes pour le réseau
    lb = np.array([t_domain[0], x_domain[0], y_domain[0]])
    ub = np.array([t_domain[1], x_domain[1], y_domain[1]])

    # Architecture du réseau
    layers = [96, 64, 32] # 24, 24, 24, 24

    # Définition des solutions exactes à tester
    exact_solutions = [
        {"name": "Impulsion gaussienne",
         "func": lambda t, x, y: exact_solution_gaussian_pulse(t, x, y, nu, beta),
         "epochs": 8000,
         "lr": 1e-4#3e-4
         },
        # {"name": "Diffusion gaussienne",
        #  "func": lambda t, x, y: exact_solution(t, x, y, nu, beta),
        #  "epochs": 5000,
        #  "lr": 4e-4
        #  }
    ]
    meshi_size = 64
    # Résultats
    results = []

    # Test avec les différentes solutions exactes
    for sol_info in exact_solutions[:1]:
        exact_sol_name = sol_info["name"]
        exact_sol_fn = sol_info["func"]
        epochs = sol_info["epochs"]
        lr = sol_info["lr"]

        print(f"\n===== Test avec la solution exacte: {exact_sol_name} =====")

        # Création du modèle PINN
        model = PINN(layers, lb, ub, nu, beta, lr=lr)

        # Génération des données d'entraînement
        t_data, x_data, y_data, u_data, t_col, x_col, y_col = generate_training_data(
            t_domain, x_domain, y_domain, exact_sol_fn, n_ic=meshi_size, n_bc=meshi_size, n_col=meshi_size)

        # Entraînement
        print(f"Entraînement du modèle pour {epochs} époques...")
        history = model.train(t_data, x_data, y_data, u_data, t_col, x_col, y_col, epochs, verbose=True)

        # Évaluation sur une grille régulière
        print("Évaluation du modèle...")
        nx, ny = 500, 500
        x_eval = np.linspace(x_domain[0], x_domain[1], nx)
        y_eval = np.linspace(y_domain[0], y_domain[1], ny)
        X_eval, Y_eval = np.meshgrid(x_eval, y_eval)

        # Évaluation à différents temps
        test_times = [i * T for i in [0.1, 0.5, 1.0]]
        for t_val in test_times:
            print(f"Évaluation à t = {t_val}")
            t_eval = np.ones((nx*ny, 1)) * t_val
            x_eval_flat = X_eval.flatten().reshape(-1, 1)
            y_eval_flat = Y_eval.flatten().reshape(-1, 1)

            # Calcul de l'erreur
            rel_l2_error, max_error, u_pred, u_exact = compute_error(
                model, t_eval, x_eval_flat, y_eval_flat, exact_sol_fn)

            print(f"Erreur L2 relative: {rel_l2_error}")
            print(f"Erreur max absolue: {max_error}")

            # Ajout des résultats
            results.append({
                "solution": exact_sol_name,
                "time": t_val,
                "rel_l2_error": rel_l2_error,
                "max_error": max_error
            })

            # Traçage des résultats
            title = f"{exact_sol_name} - Advection-diffusion 2D PINNs vs Solution exacte"
            fig = plot_solution(t_val, x_eval, y_eval, u_pred, u_exact, title)
            plt.savefig(f"solution_{exact_sol_name.replace(' ', '_')}_{t_val}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

        # Traçage de l'historique de perte
        plt.figure(figsize=(10, 6))
        plt.semilogy(history['loss'], label='Total loss')
        plt.semilogy(history['loss_data'], label='Data loss')
        plt.semilogy(history['loss_pde'], label='PDE loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Historique d\'entraînement - {exact_sol_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"loss_history_{exact_sol_name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Affichage des résultats de toutes les solutions
    print("\n===== Résumé des résultats =====")
    print("Solution exacte | Temps | Erreur L2 rel. | Erreur max abs.")
    print("-" * 60)
    for res in results:
        print(f"{res['solution']} | t={res['time']} | {res['rel_l2_error']:.6e} | {res['max_error']:.6e}")

if __name__ == "__main__":
    main()
