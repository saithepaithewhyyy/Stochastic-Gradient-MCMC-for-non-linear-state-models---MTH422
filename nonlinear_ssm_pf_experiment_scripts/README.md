# Nonlinear SSM Particle Filter Experiment Guide

This document provides a step-by-step guide for running experiments with the nonlinear state space model particle filters.

## For Synthetic Data Experiments

### 1. Running demo_setup.py

First, you need to run the appropriate demo_setup.py script for your model (LGSSM, SVM, or GARCH). Each model has its own setup script in its respective directory:

```
python nonlinear_ssm_pf_experiment_scripts/lgssm/demo_setup.py  # For LGSSM models
python nonlinear_ssm_pf_experiment_scripts/svm/demo_setup.py    # For SVM models
python nonlinear_ssm_pf_experiment_scripts/garch/demo_setup.py  # For GARCH models
```

Before running the setup script, make sure to:
- Update the `project_root` variable in the demo_setup.py file to match your project directory path
- Optionally modify other parameters like:
  - `experiment_name`: Name for this experiment run
  - `buffer_length`: Controls subsequence buffering:
    - 0: No buffering (full configuration)
    - 10: Number of buffer subsequences (buffered configuration) 
    - -1: Use all data as single subsequence (no configuration)
  - Data size and other sampler arguments

The script will create a setup.sh file in the specified experiment folder (usually ./scratch/[experiment_name]/scripts/setup.sh).

### 2. Running the Setup Script

After generating the setup script, run it to generate training and test data, initializations, and other necessary scripts:

```
./scratch/[experiment_name]/scripts/setup/setup_script.sh
```

This will create the folder structure for your experiment and generate all the necessary script files.

### 3. Running the Experiment Scripts

After setup is complete, run the following scripts in sequence to execute the full experimental pipeline:

#### a. Fit models
```
./scratch/[experiment_name]/scripts/fit/fit_script.sh
```
This runs the model fitting process and outputs results to `./scratch/[experiment_name]/out/fit/`.

#### b. Evaluate on training data
```
./scratch/[experiment_name]/scripts/eval_train/eval_train_script.sh
```
This evaluates the fitted models on training data.

#### c. Evaluate on test data
```
./scratch/[experiment_name]/scripts/eval_test/eval_test_script.sh
```
This evaluates the fitted models on test data.

#### d. Process outputs
```
./scratch/[experiment_name]/scripts/process_out/process_out_script.sh
```
This aggregates results to `./scratch/[experiment_name]/processed/`.

#### e. Evaluate trace
```
./scratch/[experiment_name]/scripts/trace_eval/trace_eval_script.sh
```
This performs trace evaluation (e.g., KSD calculations). If Reference values are not available, you may need to modify this file by removing the kstest argument and only running with ksd argument: Change `--trace_eval "kstest,ksd"` to `--trace_eval "ksd"` everywhere in `trace_eval_script.sh`

#### f. Make plots
```
./scratch/[experiment_name]/scripts/make_plots/make_plots_script.sh
```
This generates plots from the processed results.

## Important Notes

### Path Separators on Windows

Windows systems use backslashes (`\`) for file paths, while Unix/Linux systems use forward slashes (`/`). If running on Windows:

1. The scripts may contain backslashes that need to be converted to forward slashes when executing in Git Bash or similar Unix-like environments.
2. You may need to edit the script files to convert backslashes to forward slashes:
   - In run_all.sh and clear_all.sh
   - In each of the individual script files like fit_script.sh, eval_train_script.sh, etc.

Example of a path with proper forward slashes:
```
python nonlinear_ssm_pf_experiment_scripts/svm/driver.py --experiment_folder "scratch/svm_demo_nobuf" --fit
```

### Script Execution Issues

If you encounter issues with script execution:
1. Ensure that all script files have execute permission (`chmod +x script.sh`)
2. Check that line endings are in the correct format (LF, not CRLF)
3. Verify that paths in scripts use forward slashes, especially when running on Windows systems with Git Bash

### Data and Results

- Training and test data are stored in `./scratch/[experiment_name]/in/`
- Results are stored in `./scratch/[experiment_name]/out/`
- Processed results are stored in `./scratch/[experiment_name]/processed/`
- Figures are saved to `./scratch/[experiment_name]/fig/` 

## For the Gradient Error Visualizations

- Set the working directory as the main folder.
- Run the codes from nonlinear_ssm_pf_experiment_scripts/gradient_error_fig_scripts
- ⁠The results, Buffer vs Bias(Log Scaled) will be saved in scratch as png files.

