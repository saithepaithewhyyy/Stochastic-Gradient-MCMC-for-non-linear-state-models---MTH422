import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from sgmcmc_ssm.trace_metric_functions import compute_KSD
from sgmcmc_ssm.models.svm import SeqSVMSampler
from sgmcmc_ssm.models.garch import SeqGARCHSampler


# Define the Evaluator class to match what we saved in our pickle files
class Evaluator:
    def __init__(self, parameters_list, sampler):
        self.parameters_list = parameters_list
        self.sampler = sampler


def load_model_evaluator(model_type, result_dir):
    """Load model evaluator from a previous run.

    Args:
        model_type: 'svm' or 'garch'
        result_dir: directory containing evaluator pickle files

    Returns:
        tuple of (sgld_evaluator, ld_evaluator, sampler)
    """
    # Load evaluators
    with open(f"{result_dir}/sgld_evaluator.pkl", "rb") as f:
        sgld_evaluator = pickle.load(f)

    with open(f"{result_dir}/ld_evaluator.pkl", "rb") as f:
        ld_evaluator = pickle.load(f)

    # Extract sampler
    sampler = sgld_evaluator.sampler

    return sgld_evaluator, ld_evaluator, sampler


def calculate_ksd(parameters_list, model_type, data, burnin=0.33):
    """Calculate KSD metrics for parameters.

    Args:
        parameters_list: List of model parameters
        model_type: 'svm' or 'garch'
        data: Data used for model training
        burnin: Fraction of initial samples to discard

    Returns:
        dict of KSD metrics
    """
    # Create sampler based on model type
    if model_type == 'svm':
        sampler = SeqSVMSampler(n=1, m=1, observations=data)
        variables = ['A', 'Q', 'R']
    elif model_type == 'garch':
        sampler = SeqGARCHSampler(n=1, m=1, observations=data)
        variables = ['log_mu', 'logit_phi', 'logit_lambduh', 'LRinv']
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Initialize sampler
    sampler.prior_init()

    # Prepare parameter list
    param_df = pd.DataFrame({"parameters": parameters_list})

    # Apply burnin
    burnin_idx = int(len(param_df) * burnin)
    param_df = param_df.iloc[burnin_idx:]

    # Calculate gradients
    gradients = []
    for param in tqdm(param_df['parameters'], desc="Calculating gradients"):
        sampler.parameters = param
        # Calculate gradient of log joint probability
        grad = sampler.noisy_gradient(kind='pf', pf='paris', N=10000)
        gradients.append(grad)

    param_df['grad'] = gradients

    # Convert parameters to expected format for KSD calculation
    formatted_params = []
    formatted_grads = []

    for param, grad in zip(param_df['parameters'], param_df['grad']):
        if model_type == 'svm':
            param_dict = {
                'A': param.A,
                'Q': param.Q,
                'R': param.R
            }
            grad_dict = {
                'A': grad['A'],
                'Q': grad['Q'],
                'R': grad['R']
            }
        elif model_type == 'garch':
            # Extract attributes manually
            # GARCH parameters have different attributes than what compute_KSD expects
            # We need to create dictionaries with the expected attribute names
            if hasattr(param, 'log_mu'):
                param_dict = {
                    'log_mu': getattr(param, 'log_mu'),
                    'logit_phi': getattr(param, 'logit_phi'),
                    'logit_lambduh': getattr(param, 'logit_lambduh'),
                    'LRinv': getattr(param, 'LRinv')
                }
                grad_dict = {
                    'log_mu': grad['log_mu'],
                    'logit_phi': grad['logit_phi'],
                    'logit_lambduh': grad['logit_lambduh'],
                    'LRinv': grad['LRinv_vec']
                }
            else:
                # If the parameters don't have the expected attributes,
                # we'll create dictionary objects that could be used by compute_KSD
                class DictWrapper:
                    def __init__(self, d):
                        self.__dict__.update(d)

                # Create wrapper objects with the right attributes
                param_dict = DictWrapper({
                    'log_mu': 0.0,
                    'logit_phi': 0.0,
                    'logit_lambduh': 0.0,
                    'LRinv': 0.0
                })
                grad_dict = {
                    'log_mu': np.array([0.0]),
                    'logit_phi': np.array([0.0]),
                    'logit_lambduh': np.array([0.0]),
                    'LRinv': np.array([0.0])
                }

                # Skip this parameter in KSD calculation
                print("Warning: Parameter doesn't have the expected attributes "
                      "for KSD calculation")
                continue

        formatted_params.append(param_dict)
        formatted_grads.append(grad_dict)

    if len(formatted_params) == 0:
        print("Error: No valid parameters for KSD calculation.")
        print("Please check that your parameter objects have the required attributes:")
        print(f"Required attributes for {model_type}: {variables}")
        print("You may need to regenerate parameter samples with the correct "
              "format.")
        return {}

    # Compute KSD for each parameter
    result_dict = compute_KSD(
        formatted_params, formatted_grads, variables=variables)

    # Calculate log KSD
    for var in variables:
        if var in result_dict:
            result_dict[f"log_{var}"] = np.log(result_dict[var])

    return result_dict


def main():
    parser = argparse.ArgumentParser(
        description='Calculate KSD metrics from parameter traces')
    parser.add_argument('--model_type', type=str, choices=['svm', 'garch'],
                        required=True, help='Model type (svm or garch)')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory containing model results')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file to save KSD metrics (default: '
                             'ksd_metrics.csv in result_dir)')

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"{args.result_dir}/ksd_metrics.csv"

    # Load evaluators and sampler
    sgld_evaluator, ld_evaluator, sampler = load_model_evaluator(
        args.model_type, args.result_dir)

    # Get data
    data = sampler.observations

    # Calculate KSD for SGLD parameters
    print("Calculating KSD for SGLD parameters...")
    sgld_ksd = calculate_ksd(
        sgld_evaluator.parameters_list, args.model_type, data)

    # Calculate KSD for LD parameters
    print("Calculating KSD for LD parameters...")
    ld_ksd = calculate_ksd(ld_evaluator.parameters_list, args.model_type, data)

    # Combine results
    results = []
    for method, ksd_dict in [("SGLD", sgld_ksd), ("LD", ld_ksd)]:
        for var, value in ksd_dict.items():
            results.append({
                "method": method,
                "variable": var,
                "value": value
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False)
    print(f"KSD metrics saved to {args.output_file}")

    # Print summary table
    print("\nKSD Summary:")
    pivot_df = df.pivot(index='method', columns='variable', values='value')
    print(pivot_df)

    # Create summary plots
    plt.figure(figsize=(10, 6))

    var_list = list(
        set([v for v in df['variable'] if not v.startswith('log_')]))
    for i, var in enumerate(var_list):
        plt.subplot(len(var_list), 1, i+1)
        sgld_val = df[(df['method'] == 'SGLD') & (
            df['variable'] == var)]['value'].values[0]
        ld_val = df[(df['method'] == 'LD') & (
            df['variable'] == var)]['value'].values[0]
        plt.bar(['SGLD', 'LD'], [sgld_val, ld_val])
        plt.title(f"KSD for {var}")
        plt.tight_layout()

    plt.savefig(f"{args.result_dir}/ksd_summary.png", dpi=300)
    print(f"KSD summary plot saved to {args.result_dir}/ksd_summary.png")


if __name__ == "__main__":
    main()
