# Exchange Rate Data Analysis with SGMCMC

This directory contains implementations of Stochastic Gradient Markov Chain Monte Carlo (SGMCMC) methods for analyzing exchange rate data using Stochastic Volatility Model (SVM) and GARCH models.

## Overview

The code in this directory implements the experiments from the paper "Stochastic Gradient MCMC with Repulsive Forces" for exchange rate data analysis. Two models are implemented:

1. **SVM (Stochastic Volatility Model)**: A state-space model that models volatility as a latent process
2. **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**: A time series model that accounts for volatility clustering

## Data Preparation

The scripts use EUR/GBP and EUR/USD exchange rate data. To process the raw data:

```bash
python demo/exchange_rate/process_gbp_data.py
```

This creates the processed data files in the `data/` directory that other scripts will use.

## Main Workflow

### 1. Generate Parameter Samples

To generate parameter samples using SGLD (Stochastic Gradient Langevin Dynamics) and LD (Langevin Dynamics):

```bash
python demo/exchange_rate/save_garch_params.py  # For GARCH model
python demo/exchange_rate/save_svm_params.py    # For SVM model
```

**Important**: Each script takes approximately 8 hours to run with the default settings. These settings are necessary to reproduce the results from the paper.

The scripts create output directories:

- `eurgbp_garch_results/` for GARCH model results
- `eurgbp_svm_results/` for SVM model results

### 2. Calculate KSD Metrics

After generating parameter samples, calculate the Kernelized Stein Divergence:

```bash
python demo/exchange_rate/calculate_ksd.py --model_type garch --result_dir eurgbp_garch_results
python demo/exchange_rate/calculate_ksd.py --model_type svm --result_dir eurgbp_svm_results
```

KSD calculation takes approximately 1-2 hours per model, depending on the number of samples.

### 3. Fix Pickling Issues (If Needed)

If you encounter pickling errors during KSD calculation, run:

```bash
# Install dill first
pip install dill

# Then run the fix script
python demo/exchange_rate/fix_garch_evaluator.py --model_type garch --result_dir eurgbp_garch_results
python demo/exchange_rate/fix_garch_evaluator.py --model_type svm --result_dir eurgbp_svm_results
```

## File Descriptions

### Core Scripts

- **process_gbp_data.py**: Processes raw exchange rate data into a format usable by the models
- **save_garch_params.py**: Generates parameter samples for the GARCH model using SGLD and LD
- **save_svm_params.py**: Generates parameter samples for the SVM model using SGLD and LD
- **calculate_ksd.py**: Calculates KSD metrics to evaluate sample quality
- **fix_garch_evaluator.py**: Utility script to fix pickling issues with evaluator objects

### Demo Scripts

- **garch_exchange_rate_full_demo.py**: Full demo of GARCH model on exchange rate data
- **exchange_rate_full_demo.py**: Full demo of SVM model on exchange rate data
- **garch_exchange_rate_subset_demo.py**: Demo using subset of data for faster experimentation
- **exchange_rate_subset_demo.py**: Demo using subset of data for faster experimentation
- **garch_exchange_rate_single_demo.py**: Demo using a single segment of data
- **exchange_rate_single_demo.py**: Demo using a single segment of data
- **garch_eurus_ksd.py**: Script for EURUS dataset with KSD evaluation

## Expected Results

After running the full pipeline, you'll have:

1. Parameter samples in the respective results directories
2. KSD metrics in CSV format (e.g., `eurgbp_garch_results/ksd_metrics.csv`)
3. Summary plots of KSD metrics (e.g., `eurgbp_garch_results/ksd_summary.png`)

## Computation Time

Due to the computational demands of particle filtering with N=10,000 particles and the minimum 8-hour runtime required per component, one needs access to an extremely powerful machine to run this code in a reasonable timeframe.
