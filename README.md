# Stochastic Gradient MCMC for Nonlinear SSMs

This repo contains the python code for stochastic gradient MCMC in state space models for the following paper: ["Stochastic Gradient MCMC for Nonlinear SSMs"](https://arxiv.org/abs/1901.10568).

## Overview

The `sgmcmc_ssm` folder contains the python module code.

The `nonlinear_ssm_pf_experiment_scripts` folder contains python scripts that reproduce the results for the synthetic data experiments.

The `demo` folder contains python scripts that can reproduce the results for the exchange rate data experiments.

These scripts must be run with `sgmcmc_ssm` on the PYTHONPATH.
For example, running `python demo/<script name>.py` from this project root folder.
These are used to reproduce the results for exchange rate data.

Detailed instructions for running the codes are in their respective folders.

## Installation

Add the `sgmcmc_ssm` folder to the PYTHONPATH by (i) running code from the project root folder, or, (ii) adding the project root folder to the python path using `sys.path.append(<path_to_sgmcmc_ssm_code>)`

Environment Setup:
1. Create a new Python environment (recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Linux/Mac
   # OR
   .\env\Scripts\activate  # On Windows
   ```

2. Install required packages using requirements.txt:
   ```
   pip install -r requirements.txt
   ```

This will install all dependencies with the specified versions:
- numpy==1.26.4
- pandas>=2.2.3 
- scipy>=1.15.2
- seaborn>=0.13.2
- joblib>=1.3.2
- scikit-learn>=1.6.1
- tqdm>=4.66.4
- dill==0.4.0
- matplotlib>=3.7.3

Python 3.10.16 was used in our analysis on a Windows Machine with WSL.