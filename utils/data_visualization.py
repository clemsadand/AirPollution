import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats
import argparse
import os

#***********************************************************
exp_dir = "experimental_results/figures"
parser = argparse.ArgumentParser(description="PINN experiment.")
parser.add_argument('--exp_dir', type=str, default=exp_dir, help='Path of the experiment results')

args = parser.parse_args()
exp_dir = args.exp_dir

os.makedirs(exp_dir, exist_ok=True)
#*********************************************
# Set style for publication-quality figures
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (10, 8),
    'lines.linewidth': 2,
    'grid.alpha': 0.3
})

# Load the data
df_crbe = pd.read_csv(f"experimental_results/crbe/df_crbe_training_results.csv")
df_pinn = pd.read_csv(f"experimental_results/pinn/df_pinn_training_results.csv")
df_sensitivity = pd.read_csv(f"experimental_results/sensibility/df_sensitivity_data.csv")
df_runtime = pd.read_csv(f"experimental_results/fixed_runtime/fixed_runtime_comparison.csv")


# Figure 1: Convergence Analysis (L2 and L∞ errors vs mesh size)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# L2 Error convergence
ax1.loglog(df_crbe['mesh_size'], df_crbe['rel_l2_error'], 'o-', 
           label='CR-BE', color='blue', markersize=8, linewidth=3)
ax1.loglog(df_pinn['mesh_size'], df_pinn['rel_l2_error'], 's--', 
           label='PINN', color='orange', markersize=8, linewidth=3)

ax1.set_xlabel('Mesh Size')
ax1.set_ylabel('Relative L² Error')
ax1.set_title('Convergence Analysis: L² Error')
ax1.grid(True, which="both", ls="--", alpha=0.3)




# Add theoretical convergence lines
mesh_range = np.array([4, 128])
# CR-BE theoretical O(h^1.37) line
crbe_theory = 10 * (mesh_range/4)**(-1.37)
ax1.loglog(mesh_range, crbe_theory, '-.', color='blue', 
           label='$O(h^{1.37}$)', linewidth=1.5)
ax1.legend(frameon=True, fancybox=True, shadow=True)

# L∞ Error convergence
ax2.loglog(df_crbe['mesh_size'], df_crbe['max_error'], 'o-', 
           label='CR-BE', color='blue', markersize=8, linewidth=3)
ax2.loglog(df_pinn['mesh_size'], df_pinn['max_error'], 's--', 
           label='PINN', color='orange', markersize=8, linewidth=3)

ax2.set_xlabel('Mesh Size')
ax2.set_ylabel('Maximum Error (L∞)')
ax2.set_title('Convergence Analysis: L∞ Error')
ax2.grid(True, which="both", ls="--", alpha=0.3)


# Add theoretical convergence line for CR-BE
crbe_theory_linf = 0.5 * (mesh_range/4)**(-0.98)
ax2.loglog(mesh_range, crbe_theory_linf, '-.', color='blue', 
           label='$O(h^{0.98})$', linewidth=1.5)
ax2.legend(frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(f'{exp_dir}/convergence_analysis.pdf', dpi=600, bbox_inches='tight')
plt.show()

# Figure 2: Computational Efficiency Comparison
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Training time comparison
x = np.arange(len(df_crbe['mesh_size']))
width = 0.35

bars1 = ax1.bar(x - width/2, df_crbe['train_time'], width, 
                label='CR-BE', color='blue')
bars2 = ax1.bar(x + width/2, df_pinn['train_time'], width, 
                label='PINN', color='orange')

ax1.set_xlabel('Mesh Size')
ax1.set_ylabel('Training Time (seconds)')
ax1.set_title('Training Time Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(df_crbe['mesh_size'])
ax1.set_yscale('log')
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, which="both", ls="--", alpha=0.3)
ax1.tick_params(axis='both', which='major', labelsize=12)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height*1.1,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height*1.1,
             f'{height:.0f}', ha='center', va='bottom', fontsize=9)

# Efficiency metric (Error × Time)
crbe_efficiency = df_crbe['rel_l2_error'] * df_crbe['train_time']
pinn_efficiency = df_pinn['rel_l2_error'] * df_pinn['train_time']

ax2.semilogy(df_crbe['mesh_size'], crbe_efficiency, 'o-', 
             label='CR-BE', color='blue', linewidth=4, markersize=10,
           markeredgecolor='white', markeredgewidth=2)
ax2.semilogy(df_pinn['mesh_size'], pinn_efficiency, 's--', 
             label='PINN', color='orange', linewidth=4, markersize=10,
           markeredgecolor='white', markeredgewidth=2)

ax2.set_xlabel('Mesh Size')
ax2.set_ylabel('Efficiency (L² Error × Time)')
ax2.set_title('Computational Efficiency')
ax2.legend(frameon=True, fancybox=True, shadow=True)
ax2.grid(True, which="both", ls="--", alpha=0.3)
ax2.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig(f'{exp_dir}/computational_efficiency.pdf', dpi=600, bbox_inches='tight')
plt.show()


# Figure 3: Sensitivity Analysis
fig3, ax = plt.subplots(1, 1, figsize=(10, 7))
plt.semilogx(df_sensitivity['diffusion_coef'], df_sensitivity['cr_l2_error'], 'o-', 
             linewidth=3, markersize=8, label='CRBE', 
             color='blue', markeredgecolor='white', markeredgewidth=2)
plt.semilogx(df_sensitivity['diffusion_coef'], df_sensitivity['pinn_l2_error'], 's-', 
             linewidth=3, markersize=8, label='PINN', 
             color='orange', markeredgecolor='white', markeredgewidth=2)

plt.xlabel('Diffusion Coefficient')#, fontsize=16, fontweight='bold')
plt.ylabel('Relative L² Error')#, fontsize=16, fontweight='bold')
plt.title('Sensitivity to Diffusion Coefficient')#, fontsize=18, fontweight='bold', pad=20)
plt.legend(frameon=True, fancybox=True, shadow=True)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=12)

# # Add vertical line at the baseline diffusion coefficient
# plt.axvline(x=0.1, color='gray', linestyle='--', alpha=0.7, linewidth=2)
# plt.text(0.1, max(df_sensitivity['pinn_l2_error'])*0.8, 'Baseline', 
#          rotation=90, verticalalignment='bottom', fontsize=12)

plt.tight_layout()
plt.savefig(f'{exp_dir}/sensitivity_analysis.pdf', dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()


# Create a detailed memory comparison figure
fig5, ax = plt.subplots(1, 1, figsize=(12, 8))

# Create bar chart comparing memory usage
mesh_sizes = df_crbe['mesh_size'].values
x = np.arange(len(mesh_sizes))
width = 0.35

# CRBE only uses CPU memory
crbe_cpu = df_crbe['cpu_memory_usage_MB'].values
# PINN uses both GPU and CPU memory
pinn_gpu = df_pinn['gpu_memory_usage_MB'].values
pinn_cpu = df_pinn['cpu_memory_usage_MB'].values

bars1 = ax.bar(x - width/2, crbe_cpu, width, label='CRBE (CPU)', 
               color='blue', edgecolor='white', linewidth=1)
bars2 = ax.bar(x + width/2, pinn_gpu, width, label='PINN (GPU)', 
               color='orange', edgecolor='white', linewidth=1)
# bars3 = ax.bar(x + width/2, pinn_cpu, width, bottom=pinn_gpu, label='PINN (CPU)', 
#                color='orange', alpha=0.5, edgecolor='white', linewidth=1)

ax.set_xlabel('Mesh Size')#, fontsize=14, fontweight='bold')
ax.set_ylabel('Memory Usage (MB)')#, fontsize=14, fontweight='bold')
ax.set_title('Memory Usage Comparison: CPU vs GPU Implementation')#, fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(mesh_sizes)
ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
ax.set_yscale('log')
ax.grid(True, which="both", ls="--", alpha=0.3, axis='y')

# Add annotations showing total memory for PINN
for i, (gpu, cpu) in enumerate(zip(pinn_gpu, pinn_cpu)):
    total = gpu# + cpu
    if total > 0:  # Only annotate non-zero values
        ax.annotate(f'{total:.0f} MB',#\n(total)', 
                   (i + width/2, total), 
                   xytext=(0, 5), 
                   textcoords='offset points', 
                   ha='center', va='bottom', 
                   fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.7))
# Add annotations showing total memory for PINN
for i, cpu in enumerate(crbe_cpu):
    total = cpu
    if total > 0:  # Only annotate non-zero values
        ax.annotate(f'{total:.0f} MB',#\n(total)', 
                   (i - width/2, total), 
                   xytext=(0, 5), 
                   textcoords='offset points', 
                   ha='center', va='bottom', 
                   fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{exp_dir}/memory_comparison_cpu_gpu.pdf', dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# Figure 5: Fixed Runtime Budget Analysis
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Extract runtime budget data
pinn_runtime = df_runtime[df_runtime['method'] == 'PINN']
crbe_runtime = df_runtime[df_runtime['method'] == 'CRBE']

# Group by time budget and calculate means
pinn_grouped = pinn_runtime.groupby('time_budget').agg({
    'rel_l2_error': 'mean',
    'max_error': 'mean',
    'epochs_completed': 'mean'
}).reset_index()

crbe_grouped = crbe_runtime.groupby('time_budget').agg({
    'rel_l2_error': 'mean',
    'max_error': 'mean'
}).reset_index()

# L2 Error vs Time Budget
ax1.plot(pinn_grouped['time_budget'], pinn_grouped['rel_l2_error'], 's-', 
         label='PINN', color='orange', markersize=8, linewidth=3)
ax1.axhline(y=crbe_grouped['rel_l2_error'].iloc[0], color='blue', 
            linestyle='-', linewidth=3, label='CR-BE (constant)')

ax1.set_xlabel('Time Budget (seconds)')
ax1.set_ylabel('Relative L² Error')
ax1.set_title('Performance vs Time Budget')
ax1.legend(frameon=True, fancybox=True, shadow=True)
ax1.grid(True, which="both", ls="--", alpha=0.3)

# Epochs completed by PINN
ax2.plot(pinn_grouped['time_budget'], pinn_grouped['epochs_completed'], 'o-', 
         color='green', markersize=8, linewidth=3)
ax2.set_xlabel('Time Budget (seconds)')
ax2.set_ylabel('Epochs Completed')
ax2.set_title('PINN Training Progress')
ax2.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig(f'{exp_dir}/runtime_budget_analysis.pdf', dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

