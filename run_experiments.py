#!/usr/bin/env python3
import subprocess
import sys
import os

epochs = 1
print("Running PINN experiments...")
subprocess.run([sys.executable, "-m", "experiments.pinn_experiments", 
                   "--width=4", f"--epochs={epochs}", "--activation=tanh"])
                   
print("Running CRBE experiments...")
subprocess.run([sys.executable, "-m", "experiments.crbe_experiments"])
    
print("Running sensitivity analysis...")
subprocess.run([sys.executable, "-m", "experiments.sensitivity_analysis", "--width=4", f"--epochs={epochs}", "--activation=tanh"])
    
print("Running fixed runtime experiments...")
subprocess.run([sys.executable, "-m", "experiments.fixed_runtime_experiments", "--run_for_testing=True"])

print("Generating visualizations...")
subprocess.run([sys.executable, "-m", "utils.data_visualization"])
    
print("Generating LaTeX tables...")
subprocess.run([sys.executable, "-m", "utils.table_generator"])
    
print("\nAll experiments completed!")
print("Results saved in experimental_results/")

