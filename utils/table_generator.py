import torch
import pandas as pd
import numpy as np
from scipy.stats import linregress
import argparse
import os
#***********************************************************
exp_dir = "experimental_results/tables"
parser = argparse.ArgumentParser(description="PINN experiment.")
parser.add_argument('--exp_dir', type=str, default=exp_dir, help='Path of the experiment results')

args = parser.parse_args()
exp_dir = args.exp_dir

os.makedirs(exp_dir, exist_ok=True)
#***********************************************************

def format_sci(x):
    from math import log10, floor

    if x == 0:
        return "$0$"

    abs_x = abs(x)

    # Scientific notation for very small or very large numbers
    if abs_x < 1e-4 or abs_x >= 1e4:
        s = f"{x:.5e}"  # scientific with 5 digits after decimal
        base, exp = s.split('e')
        base = f"{float(base):.5f}".rstrip('0').rstrip('.')  # remove trailing zeros
        return f"${base[:4]}\\cdot 10^{{{int(exp)}}}$"
    else:
        # Count digits before decimal
        int_part = int(abs_x)
        digits_before_dot = len(str(int_part))
        
        if digits_before_dot >= 4:
            return f"${x:.1f}$"  # Round to 1 decimals
        elif digits_before_dot >= 3:
            return f"${x:.2f}$"  # Round to 2 decimals
        elif digits_before_dot >= 2:
            return f"${x:.3f}$"  # Round to 3 decimals
        else:
            return f"${x:.4f}$"  # Round to 4 decimals




def generate_latex_tables(df_crbe, df_pinn, memory_data=None, sensitivity_data=None, df_fixed_runtime=None):
    """Generate LaTeX tables from DataFrame results."""
    tables = {}

    mesh_sizes = df_crbe['mesh_size'].values

    log_h_crbe = np.log(1 / df_crbe['mesh_size'].values)
    log_err_l2_crbe = np.log(df_crbe['rel_l2_error'].values)
    log_err_linf_crbe = np.log(df_crbe['max_error'].values)

    log_h_pinn = np.log(1 / df_pinn['mesh_size'].values)
    log_err_l2_pinn = np.log(df_pinn['rel_l2_error'].values)
    log_err_linf_pinn = np.log(df_pinn['max_error'].values)

    crbe_l2_rate, _, crbe_l2_r2, _, _ = linregress(log_h_crbe, log_err_l2_crbe)
    crbe_linf_rate, _, crbe_linf_r2, _, _ = linregress(log_h_crbe, log_err_linf_crbe)
    pinn_l2_rate, _, pinn_l2_r2, _, _ = linregress(log_h_pinn, log_err_l2_pinn)
    pinn_linf_rate, _, pinn_linf_r2, _, _ = linregress(log_h_pinn, log_err_linf_pinn)

    # Table 1: Convergence comparison
    table1 = "\\begin{table}[htbp]\n\\centering\n"
    table1 += "\\caption{Convergence comparison of CR-BE and PINN methods}\n"
    table1 += "\\label{tab:convergence_comparison}\n"
    table1 += "\\begin{tabular}{ccccccc}\n\\toprule\n"
    table1 += "\\multirow{2}{*}{Mesh Size} & \\multicolumn{2}{c}{Relative $L^2$ Error} & "
    table1 += "\\multicolumn{2}{c}{Maximum Error ($L^\\infty$)} & \\multicolumn{2}{c}{Training Time (s)} \\\\\n"
    table1 += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n"
    table1 += "& CR-BE & PINN & CR-BE & PINN & CR-BE & PINN \\\\\n\\midrule\n\\midrule\n"
    
    for i, mesh in enumerate(mesh_sizes):
        cr_l2 = format_sci(df_crbe['rel_l2_error'].iloc[i])
        pinn_l2 = format_sci(df_pinn['rel_l2_error'].iloc[i])
        cr_linf = format_sci(df_crbe['max_error'].iloc[i])
        pinn_linf = format_sci(df_pinn['max_error'].iloc[i])
        cr_time = f"${df_crbe['train_time'].iloc[i]:.2f}$"
        pinn_time = f"${df_pinn['train_time'].iloc[i]:.2f}$"
        
        table1 += f"{mesh} & {cr_l2} & {pinn_l2} & {cr_linf} & {pinn_linf} & {cr_time} & {pinn_time} \\\\\n"
    
    table1 += "\\bottomrule\n\\end{tabular}\n\\end{table}"

    # Table 2: Convergence rates
    table2 = "\\begin{table}[htbp]\n\\centering\n"
    table2 += "\\caption{Empirical convergence rates for CR-BE and PINN methods}\n"
    table2 += "\\label{tab:convergence_rates}\n"
    table2 += "\\begin{tabular}{ccccc}\n\\toprule\n"
    table2 += "\\multirow{2}{*}{Method} & \\multicolumn{2}{c}{Convergence Rate} & "
    table2 += "\\multicolumn{2}{c}{Goodness of Fit ($R^2$)} \\\\\n"
    table2 += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
    table2 += "& $L^2$ Error & $L^\\infty$ Error & $L^2$ Error & $L^\\infty$ Error \\\\\n\\midrule\n\\midrule\n"
    
    table2 += f"CR-BE & ${crbe_l2_rate:.4f}$ & ${crbe_linf_rate:.4f}$ & "
    table2 += f"${crbe_l2_r2:.4f}$ & ${crbe_linf_r2:.4f}$ \\\\\n"
    
    table2 += f"PINN & ${pinn_l2_rate:.4f}$ & ${pinn_linf_rate:.4f}$ & "
    table2 += f"${pinn_l2_r2:.4f}$ & ${pinn_linf_r2:.4f}$ \\\\\n"
    
    table2 += "\\bottomrule\n\\end{tabular}\n\\end{table}"


    # Table 3: Computational resources (if memory data available)
    table3 = "\\begin{table}[htbp]\n\\centering\n"
    table3 += "\\caption{Computational resource requirements}\n"
    table3 += "\\label{tab:computational_resources}\n"
    table3 += "\\begin{tabular}{ccccc}\n\\toprule\n"
    table3 += "\\multirow{2}{*}{Mesh Size} & \\multicolumn{2}{c}{Memory Usage (MB)} & "
    table3 += "\\multicolumn{2}{c}{DOFs / Parameters} \\\\\n"
    table3 += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
    table3 += "& CR-BE & PINN & CR-BE & PINN \\\\\n\\midrule\n\\midrule\n"

    # If we have memory data
    if memory_data is not None:
        for i, mesh in enumerate(mesh_sizes):
            mem_crbe = f"{format_sci(memory_data['cr_memory_mb'].iloc[i])}"
            mem_pinn = f"{format_sci(memory_data['pinn_memory_mb'].iloc[i])}"
            dofs_crbe = f"${df_crbe['number_of_collocation_points'].iloc[i]}$"
            
            # Check if we have parameter counts for PINN
            if 'n_parameters' in df_pinn.columns:
                params_pinn = f"${df_pinn['n_parameters'].iloc[i]}$"
            else:
                params_pinn = "$-$"
                
            table3 += f"{mesh} & {mem_crbe} & {mem_pinn} & {dofs_crbe} & {params_pinn} \\\\\n"
    else:
        # If no memory data, use placeholder or skip
        for i, mesh in enumerate(mesh_sizes):
            dofs_crbe = f"${df_crbe['number_of_collocation_points'].iloc[i]}$"
            
            # Check if we have parameter counts for PINN
            if 'n_parameters' in df_pinn.columns:
                params_pinn = f"${df_pinn['n_parameters'].iloc[i]}$"
            else:
                params_pinn = "$-$"
                
            table3 += f"{mesh} & $-$ & $-$ & {dofs_crbe} & {params_pinn} \\\\\n"
    table3 += "\\bottomrule\n\\end{tabular}\n\\end{table}"

    # Table 4: Efficiency comparison
    table4 = "\\begin{table}[htbp]\n\\centering\n"
    table4 += "\\caption{Efficiency comparison ($L^2$ error $\\times$ training time)}\n"
    table4 += "\\label{tab:efficiency_comparison}\n"
    table4 += "\\begin{tabular}{ccc}\n\\toprule\n"
    table4 += "Mesh Size & CR-BE Efficiency & PINN Efficiency \\\\\n\\midrule\n\\midrule\n"
    
    for i, mesh in enumerate(mesh_sizes):
        eff_crbe = df_crbe['rel_l2_error'].iloc[i] * df_crbe['train_time'].iloc[i]
        eff_pinn = df_pinn['rel_l2_error'].iloc[i] * df_pinn['train_time'].iloc[i]
        
        table4 += f"{mesh} & {format_sci(eff_crbe)} & {format_sci(eff_pinn)} \\\\\n"
    
    table4 += "\\bottomrule\n\\end{tabular}\n\\end{table}"

    # Table 5: Summary statistics
    table5 = "\\begin{table}[htbp]\n\\centering\n"
    table5 += "\\caption{Summary of method performance}\n"
    table5 += "\\label{tab:summary_statistics}\n"
    table5 += "\\begin{tabular}{lcc}\n\\toprule\n"
    table5 += "Metric & CR-BE & PINN \\\\\n\\midrule\n\\midrule\n"
    
    min_l2_crbe = format_sci(df_crbe['rel_l2_error'].min())
    min_l2_pinn = format_sci(df_pinn['rel_l2_error'].min())
    min_linf_crbe = format_sci(df_crbe['max_error'].min())
    min_linf_pinn = format_sci(df_pinn['max_error'].min())
    max_time_crbe = f"${df_crbe['train_time'].max():.2f}$"
    max_time_pinn = f"${df_pinn['train_time'].max():.2f}$"
    
    table5 += f"Minimum $L^2$ Error & {min_l2_crbe} & {min_l2_pinn} \\\\\n"
    table5 += f"Minimum $L^\\infty$ Error & {min_linf_crbe} & {min_linf_pinn} \\\\\n"
    table5 += f"Maximum Training Time (s) & {max_time_crbe} & {max_time_pinn} \\\\\n"
    table5 += f"$L^2$ Convergence Rate & {format_sci(crbe_l2_rate)} & {format_sci(pinn_l2_rate)} \\\\\n"
    table5 += f"$L^\\infty$ Convergence Rate & {format_sci(crbe_linf_rate)} & {format_sci(pinn_linf_rate)} \\\\\n"
    
    # Determine scaling based on convergence rate
    scaling_crbe = f"$O(n^{{{abs(crbe_l2_rate):.1f}}})$"
    scaling_pinn = f"$O(n^{{{abs(pinn_l2_rate):.1f}}})$"
    table5 += f"Error Scaling & {scaling_crbe} & {scaling_pinn} \\\\\n"
    
    table5 += "\\bottomrule\n\\end{tabular}\n\\end{table}"


    # Table 6: Method characteristics
    table6 = "\\begin{table}[htbp]\n\\centering\n"
    table6 += "\\caption{Quantitative evidence for method characteristics}\n"
    table6 += "\\label{tab:method_characteristics}\n"
    table6 += "\\begin{tabular}{lcc}\n\\toprule\n"
    table6 += "Characteristic & CR-BE & PINN \\\\\n\\midrule\n\\midrule\n"
    
    # Use data for mesh size 64 (index 4) as reference point
    mesh_64_idx = list(mesh_sizes).index(64) if 64 in mesh_sizes else -2  # Second to last as fallback
    
    eff_crbe_64 = df_crbe['rel_l2_error'].iloc[mesh_64_idx] * df_crbe['train_time'].iloc[mesh_64_idx]
    eff_pinn_64 = df_pinn['rel_l2_error'].iloc[mesh_64_idx] * df_pinn['train_time'].iloc[mesh_64_idx]
    
    table6 += f"Accuracy (Best $L^2$ Error) & {min_l2_crbe} & {min_l2_pinn} \\\\\n"
    table6 += f"Computational Efficiency (Time for mesh=64) & ${df_crbe['train_time'].iloc[mesh_64_idx]:.2f}$ s & ${df_pinn['train_time'].iloc[mesh_64_idx]:.2f}$ s \\\\\n"
    
    if memory_data is not None:
        table6 += f"Memory Usage (MB for mesh=64) & ${memory_data['cr_memory_mb'].iloc[mesh_64_idx]:.2f}$ & ${memory_data['pinn_memory_mb'].iloc[mesh_64_idx]:.2f}$ \\\\\n"
    else:
        table6 += f"Memory Usage (MB for mesh=64) & $-$ & $-$ \\\\\n"
        
    table6 += f"Convergence Rate ($L^2$) & ${crbe_l2_rate:.4f}$ & ${pinn_l2_rate:.4f}$ \\\\\n"
    table6 += f"Error/Cost Ratio (mesh=64) & ${eff_crbe_64:.4f}$ & ${eff_pinn_64:.4f}$ \\\\\n"
    
    table6 += "\\bottomrule\n\\end{tabular}\n\\end{table}"

    # Table 7: Parameter sensitivity (if available)
    # Table 7: Sensitivity to diffusion coefficient
    mesh_selection = [64]
    if sensitivity_data is not None:
        for mesh in mesh_selection:
            table7 = "\\begin{table}[htbp]\n\\centering\n"
            table7 += "\\caption{Sensitivity to diffusion coefficient variations}\n"
            table7 += "\\label{tab:sensitivity_diffusion}\n"
            table7 += "\\begin{tabular}{ccc}\n\\toprule\n"
            table7 += "Diffusion Coefficient & CR-BE $L^2$ Error & PINN $L^2$ Error \\\\\n\\midrule\n\\midrule\n"
            for idx, row in sensitivity_data[sensitivity_data["mesh_size"]==mesh].iterrows():
                table7 += f"${row['diffusion_coef']:.4f}$ & {format_sci(row['cr_l2_error'])} & {format_sci(row['pinn_l2_error'])} \\\\\n"
            table7 += "\\bottomrule\n\\end{tabular}\n\\end{table}"
            tables['sensitivity'] = table7

    if df_fixed_runtime is not None:
        summary_df = df_fixed_runtime.groupby(['method', 'time_budget']).agg({
            'rel_l2_error': "mean", #['mean', 'std'],
            'max_error': "mean",#['mean', 'std'],
            'actual_runtime': "mean",#['mean', 'std'],
            'epochs_completed': 'mean',
            "gpu_memory_usage_MB": "mean",
            "cpu_memory_usage_MB": "mean",
        })
        
        # summary_df.columns = columns
        summary_df.reset_index(inplace=True)
        summary_df["time_utilized"] = ((summary_df["actual_runtime"] * 100) / summary_df["time_budget"]).round(0)
        
        # Split into two DataFrames
        df_fixed_crbe = summary_df[summary_df['method'] == 'CRBE'].reset_index(drop=True)
        df_fixed_pinn = summary_df[summary_df['method'] == 'PINN'].reset_index(drop=True)
        
        table8 = r"""\begin{table}[htbp]
        \centering
        \caption{Performance comparison under fixed runtime budgets}
        \label{tab:fixed_runtime_comparison}
        \begin{tabular}{cccccccccc}
        \toprule
        \multirow{2}{*}{Time Budget(s)} & \multicolumn{2}{c}{Rel $L^2$ Error} & \multicolumn{2}{c}{Max Error ($L^\infty$)} & \multicolumn{2}{c}{Time Utilized (\%)} & \multicolumn{2}{c}{Memory Usage (MB)} & Epochs \\
        \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} 
        & CR-BE & PINN & CR-BE & PINN & CR-BE & PINN & CR-BE & PINN & (PINN) \\
        \midrule
        """
        
        for i in range(len(df_fixed_crbe)):
            row1 = df_fixed_crbe.iloc[i] 
            row2 = df_fixed_pinn.iloc[i]
            table8 += f"{row1['time_budget']} & {format_sci(row1['rel_l2_error'])} & {format_sci(row2['rel_l2_error'])} & {format_sci(row1['max_error'])} & {format_sci(row2['max_error'])} & {row1['time_utilized']} & {row2['time_utilized']} & {format_sci(row1['cpu_memory_usage_MB'])} & {format_sci(row2['gpu_memory_usage_MB'])} & {round(row2['epochs_completed'])} \\\\\n"
        
        table8 += r"""\bottomrule
        \end{tabular}
        \end{table}"""
        
        # print(table8)

    
    tables = {
        "convergence_comparison": table1,
        "convergence_rates": table2,
        "computational_resources": table3,
        "efficiency_comparison": table4,
        "summary_statistics": table5,
        "method_characteristics": table6
    }
    
    if table7:
        tables["parameter_sensitivity"] = table7
    if table8:
        tables["fixed_runtime"] = table8
        
    return tables

#==================================================
df_crbe = pd.read_csv(f"experimental_results/crbe/df_crbe_training_results.csv")
df_pinn = pd.read_csv(f"experimental_results/pinn/df_pinn_training_results.csv")
sensitivity_data = pd.read_csv(f"experimental_results/sensibility/df_sensitivity_data.csv")
df_fixed_runtime = pd.read_csv(f"experimental_resutls/fixed_runtime/fixed_runtime_comparison.csv")

memory_data = pd.DataFrame(
    {
        "cr_memory_mb": list(df_crbe["cpu_memory_usage_MB"].values),
        "pinn_memory_mb": list(df_pinn["gpu_memory_usage_MB"].values) #if torch.cuda.is_available() else list(df_pinn["cpu_memory_usage_MB"].values)
    }
)




# Generate tables from your results
tables = generate_latex_tables(df_crbe, df_pinn, memory_data=memory_data, sensitivity_data=sensitivity_data, df_fixed_runtime=df_fixed_runtime)
    
# Write tables to file
with open(f'{exp_dir}/convergence_tables.tex', 'w') as f:
    for name, table in tables.items():
        f.write(f"% {name}\n")
        f.write(table)
        f.write("\n\n")
        
print(f"LaTeX tables generated and saved to {exp_dir}/convergence_tables.tex")


