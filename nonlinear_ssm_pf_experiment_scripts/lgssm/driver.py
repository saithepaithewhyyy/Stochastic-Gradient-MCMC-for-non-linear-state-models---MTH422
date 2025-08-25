""" Experiment Driver

Call python <path_to_this_file>.py --help to see documentation
"""
from sgmcmc_ssm.models.lgssm import (
    LGSSMSampler,
    LGSSMPrior,
    LGSSMPreconditioner,
    generate_lgssm_data,
)
from sgmcmc_ssm.plotting_utils import (
    plot_metrics, plot_trace_plot,
)
from sgmcmc_ssm.driver_utils import (
    script_builder, TqdmToLogger,
    pandas_write_df_to_csv, joblib_write_to_file,
)
from sgmcmc_ssm.metric_functions import (
    sample_function_parameters,
    metric_function_parameters,
    metric_compare_x,
    noisy_logjoint_loglike_metric,
)
from sgmcmc_ssm.evaluator import (
    SamplerEvaluator, OfflineEvaluator, half_average_parameters_list,
)
import logging  # For Logs
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import argparse
import time
import joblib
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.getcwd())  # Fix Python Path


matplotlib.use('Agg')  # For Cluster

# Utility functions for path handling


def normalize_path(path):
    """Normalize a path to use the OS-appropriate path separators"""
    return os.path.normpath(path)


def safe_make_path(path):
    """Create a directory if it doesn't exist, ensuring path separators are correct"""
    path = normalize_path(path)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


LOGGING_FORMAT = '%(levelname)s: %(asctime)s - %(name)s: %(message)s ...'
logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
)
logger = logging.getLogger(name=__name__)


DEFAULT_OPTIONS = dict(
    model_type="LGSSM",
    prior_variance=100.0,
    max_num_iters=1000000,
    max_time=60,
    eval_freq=1,
    max_eval_iterations=1000,
    max_eval_time=60,
    steps_per_iteration=1,
    checkpoint_num_iters=1000,
    checkpoint_time=60*30,
)


# Script Argument Parser
def construct_parser():
    """ Define script argument parser """
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
    )

    # Key Value Args
    parser.add_argument("--experiment_folder",
                        help="path to experiment",
                        type=str,
                        )
    parser.add_argument("--experiment_id",
                        default=0,
                        help="id of experiment (optional)",
                        type=int,
                        )
    parser.add_argument("--path_to_additional_args", default="",
                        help="additional arguments to pass to setup",
                        type=str,
                        )

    # Action Args
    parser.add_argument("--setup", action='store_const', const=True,
                        help="flag for whether to setup data, inits, and fit/eval args",
                        )
    parser.add_argument("--fit", action='store_const', const=True,
                        help="flag for whether to run sampler/optimization",
                        )
    parser.add_argument("--eval", default="",
                        help="run evaluation of parameters on target data (e.g. 'train', 'test', 'half_avg_train')",
                        type=str,
                        )
    parser.add_argument("--trace_eval", default="",
                        help="run evaluation on parameter trace (e.g. 'ksd', 'kstest')",
                        type=str,
                        )
    parser.add_argument("--process_out", action='store_const', const=True,
                        help="flag for whether to aggregate output",
                        )
    parser.add_argument("--make_plots", action='store_const', const=True,
                        help="flag for whether to plot aggregated output",
                        )
    parser.add_argument("--make_scripts", action='store_const', const=True,
                        help="flag for setup to only recreate scripts",
                        )

    return parser

# Main Dispatcher


def main(experiment_folder, experiment_id, path_to_additional_args,
         setup, fit, eval, trace_eval, process_out, make_plots,
         make_scripts, **kwargs):
    """ Main Dispatcher see construct_parser for argument help """
    if kwargs:
        logger.warning("Unused kwargs: {0}".format(kwargs))
    out = {}

    if setup:
        out['setup'] = do_setup(experiment_folder, path_to_additional_args)
        make_scripts = True

    # Ensure we have an absolute path for the experiment folder
    if not os.path.isabs(experiment_folder):
        experiment_folder = os.path.abspath(experiment_folder)
    logger.info("Using absolute experiment folder path: {}".format(
        experiment_folder))

    # Normalize experiment folder path
    experiment_folder = normalize_path(experiment_folder)
    logging.info("Extracting Options for experiment id {0} from folder {1}".format(
        experiment_id, experiment_folder))
    path_to_arg_list = normalize_path(
        os.path.join(experiment_folder, "in", "options.p"))
    logger.info("Loading arguments from: {0}".format(path_to_arg_list))

    # Check if the options file exists
    if not os.path.isfile(path_to_arg_list):
        logger.error("Options file not found: {}".format(path_to_arg_list))
        # Try to find the file in a subdirectory
        for root, _, files in os.walk('.'):
            for file in files:
                if file == "options.p" and "in" in root:
                    alt_path = normalize_path(os.path.join(root, file))
                    logger.info("Found options at: {}".format(alt_path))
                    path_to_arg_list = alt_path
                    experiment_folder = normalize_path(
                        os.path.dirname(os.path.dirname(alt_path)))
                    logger.info("Updated experiment folder to: {}".format(
                        experiment_folder))
                    break

    arg_list = joblib.load(path_to_arg_list)
    experiment_options = arg_list[experiment_id]

    # Normalize all path options
    for key in ['experiment_folder', 'path_to_data', 'path_to_test_data', 'path_to_init']:
        if key in experiment_options:
            experiment_options[key] = normalize_path(experiment_options[key])

    logger.info("Experiment Options: {0}".format(experiment_options))

    if make_scripts:
        out['make_scripts'] = do_make_scripts(
            experiment_folder, path_to_additional_args, arg_list)
    if fit:
        out['fit'] = do_fit(**experiment_options)

    if eval != "":
        for eval_ in eval.split(","):
            if eval_ in ['train', 'half_avg_train', 'test', 'half_avg_test']:
                out['eval_{0}'.format(eval_)] = do_eval(
                    target=eval_,
                    **experiment_options,
                )
            else:
                raise ValueError(
                    "Unrecognized 'eval' target {0}".format(eval_))

    if trace_eval != "":
        for trace_eval_ in trace_eval.split(","):
            if trace_eval_ == "ksd":
                out['trace_eval_{0}'.format(trace_eval)] = do_eval_ksd(
                    **experiment_options,
                )
            elif trace_eval_ == "ess":
                raise NotImplementedError()
            elif trace_eval_ == "kstest":
                out['trace_eval_{0}'.format(trace_eval)] = do_eval_ks_test(
                    **experiment_options,
                )
            else:
                raise ValueError(
                    "Unrecognized 'trace_eval' target {0}".format(trace_eval_))

    if process_out:
        out['process_out'] = do_process_out(experiment_folder)

    if make_plots:
        out['make_plots'] = do_make_plots(experiment_folder)

    if len(out.keys()) == 0:
        raise ValueError("No Flags Set")

    return out

# Setup Function


def do_setup(experiment_folder, path_to_additional_args):
    """ Setup Shell Scripts for Experiment """
    additional_args = joblib.load(path_to_additional_args)

    # Setup Data
    logger.info("Setting Up Data")
    data_args = setup_train_test_data(experiment_folder, **additional_args)

    # Setup
    logger.info("Saving Experiment Options per ID")
    sampler_args = additional_args['sampler_args']
    arg_list = dict_product(sampler_args, data_args)
    options_df = setup_options(experiment_folder, arg_list)
    return options_df

# Make Scripts


def do_make_scripts(experiment_folder, path_to_additional_args, arg_list):
    additional_args = joblib.load(path_to_additional_args)
    options_df = pd.DataFrame(arg_list)

    # Setup Shell Scripts
    logger.info("Setting up Shell Scripts")
    shell_args_base = [{
        '--experiment_folder': experiment_folder,
        '--experiment_id': experiment_id,
    } for experiment_id in options_df['experiment_id']
    ]
    # Fit Script
    script_builder(
        script_name="fit",
        python_script_path=additional_args['python_script_path'],
        python_script_args=[update_dict(
            args, {"--fit": None}) for args in shell_args_base],
        path_to_shell_script=additional_args['path_to_shell_script'],
        project_root=additional_args['project_root'],
        conda_env_name=additional_args.get('conda_env_name', None),
        **additional_args.get('fit_script_kwargs', {})
    )

    # Eval Scripts
    script_builder(
        script_name="eval_train",
        python_script_path=additional_args['python_script_path'],
        python_script_args=[update_dict(args, {"--eval": 'half_avg_train'})
                            for args in shell_args_base],
        path_to_shell_script=additional_args['path_to_shell_script'],
        project_root=additional_args['project_root'],
        conda_env_name=additional_args.get('conda_env_name', None),
        **additional_args.get('eval_script_kwargs', {})
    )

    script_builder(
        script_name="eval_test",
        python_script_path=additional_args['python_script_path'],
        python_script_args=[update_dict(args, {"--eval": 'half_avg_test'})
                            for args in shell_args_base],
        path_to_shell_script=additional_args['path_to_shell_script'],
        project_root=additional_args['project_root'],
        conda_env_name=additional_args.get('conda_env_name', None),
        **additional_args.get('eval_script_kwargs', {})
    )

    script_builder(
        script_name="trace_eval",
        python_script_path=additional_args['python_script_path'],
        python_script_args=[update_dict(args, {"--trace_eval": 'kstest,ksd'})
                            for args in shell_args_base],
        path_to_shell_script=additional_args['path_to_shell_script'],
        project_root=additional_args['project_root'],
        conda_env_name=additional_args.get('conda_env_name', None),
        **additional_args.get('eval_script_kwargs', {})
    )

    # Process Script
    script_builder(
        script_name="process_out",
        python_script_path=additional_args['python_script_path'],
        python_script_args=[{
            "--experiment_folder": experiment_folder,
            "--process_out": None,
        }],
        path_to_shell_script=additional_args['path_to_shell_script'],
        project_root=additional_args['project_root'],
        conda_env_name=additional_args.get('conda_env_name', None),
        **additional_args.get('process_out_script_kwargs', {})
    )

    # Plot Script
    script_builder(
        script_name="make_plots",
        python_script_path=additional_args['python_script_path'],
        python_script_args=[{
            "--experiment_folder": experiment_folder,
            "--make_plots": None,
        }],
        path_to_shell_script=additional_args['path_to_shell_script'],
        project_root=additional_args['project_root'],
        conda_env_name=additional_args.get('conda_env_name', None),
        **additional_args.get('make_plots_script_kwargs', {})
    )

    # Run All Script
    path_to_runall_script = os.path.join(
        additional_args['path_to_shell_script'], 'run_all.sh')
    with open(path_to_runall_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("cd {0}\n".format(additional_args['project_root']))
        f.write("{0}\n".format(os.path.join(
            additional_args['path_to_shell_script'], 'fit.sh')))
        f.write("{0}\n".format(os.path.join(
            additional_args['path_to_shell_script'], 'eval_train.sh')))
        f.write("{0}\n".format(os.path.join(
            additional_args['path_to_shell_script'], 'eval_test.sh')))
        f.write("{0}\n".format(os.path.join(
            additional_args['path_to_shell_script'], 'process_out.sh')))
        f.write("{0}\n".format(os.path.join(
            additional_args['path_to_shell_script'], 'make_plots.sh')))
    os.chmod(path_to_runall_script, 0o775)
    logger.info("Run All Script at {0}".format(path_to_runall_script))

    # Clear All Script
    path_to_clear_script = os.path.join(
        additional_args['path_to_shell_script'], 'clear_all.sh')
    with open(path_to_clear_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("cd {0}\n".format(
            os.path.join(additional_args['project_root'], experiment_folder)))
        f.write("rm -r ./in ./out ./scratch ./fig\n")
        f.write("cd {0}\n".format(
            os.path.join(additional_args['project_root'],
                additional_args['path_to_shell_script'])))
        f.write(
            "rm -r ./fit ./eval_train ./eval_test ./process_out ./make_plots ./trace_eval\n")
    os.chmod(path_to_clear_script, 0o775)
    logger.info("Clear Script at {0}".format(path_to_clear_script))

    return options_df

# Fit Module


def do_fit(
        experiment_name, experiment_id,
        experiment_folder, path_to_data, path_to_init,
        model_type, prior_variance,
        inference_method, eval_freq,
        max_num_iters, steps_per_iteration, max_time,
        checkpoint_num_iters, checkpoint_time,
        **kwargs):
    """ Fit function

    Saves list of parameters + runtimes to <experiment_folder>/out/fit/

    Args:
        experiment_name, experiment_id - experiment id parameters
        experiment_folder, path_to_data, path_to_init - paths to input + output
        model_type, prior_variance - args for get_model_sampler_prior()
        inference_method - get_model_sampler_step()
        eval_freq - how frequently to eval metric funcs
        max_num_iters, steps_per_iteration, max_time - how long to fit/train
        checkpoint_num_iters, checkpoint_time - how frequent to checkpoint
        **kwargs - contains inference_method kwargs

    """
    logger.info("Beginning Experiment {0} for id:{1}".format(
        experiment_name, experiment_id))

    Sampler, Prior = get_model_sampler_prior(model_type)

    # Ensure we have absolute paths
    if not os.path.isabs(experiment_folder):
        experiment_folder = os.path.abspath(experiment_folder)
    logger.info("Using absolute experiment folder path: {}".format(
        experiment_folder))

    if not os.path.isabs(path_to_data):
        path_to_data = os.path.abspath(path_to_data)

    if not os.path.isabs(path_to_init):
        path_to_init = os.path.abspath(path_to_init)

    # Make Paths
    experiment_folder = normalize_path(experiment_folder)
    path_to_data = normalize_path(path_to_data)
    path_to_init = normalize_path(path_to_init)

    path_to_out = normalize_path(os.path.join(experiment_folder, "out", "fit"))
    path_to_fig = normalize_path(os.path.join(experiment_folder, "fig", "fit",
                                              "{0:0>4}".format(experiment_id)))
    path_to_scratch = normalize_path(
        os.path.join(experiment_folder, 'scratch'))
    path_to_fit_state = normalize_path(os.path.join(path_to_scratch,
                                                    "fit_{0:0>4}_state.p".format(experiment_id)))

    # Log the paths for debugging
    logger.info("Data path: {0}".format(path_to_data))
    logger.info("Init path: {0}".format(path_to_init))
    logger.info("Output path: {0}".format(path_to_out))
    logger.info("Figure path: {0}".format(path_to_fig))
    logger.info("Scratch path: {0}".format(path_to_scratch))
    logger.info("Fit state path: {0}".format(path_to_fit_state))

    safe_make_path(path_to_out)
    safe_make_path(path_to_fig)
    safe_make_path(path_to_scratch)

    # Load Train Data
    logger.info("Getting Data at {0}".format(path_to_data))
    data = joblib.load(path_to_data)
    observations = data['observations']

    # Set Metric + Sample Functions for Evaluator
    parameter_names = ['A', 'C', 'Q', 'R']
    sample_functions = [sample_function_parameters(parameter_names)]
    metric_functions = [noisy_logjoint_loglike_metric()]
    if 'parameters' in data.keys():
        metric_functions += [
            metric_function_parameters(
                parameter_names=parameter_names,
                target_values=[getattr(data['parameters'], parameter_name)
                               for parameter_name in parameter_names],
                metric_names=['logmse' for _ in parameter_names],
            )
        ]

    # Check if existing sampler and evaluator state exists
    if os.path.isfile(path_to_fit_state):
        logger.info("Continuing Evaluation from {0}".format(path_to_fit_state))
        fit_state = joblib.load(path_to_fit_state)
        init_parameters = fit_state['parameters']
        parameters_list = fit_state['parameters_list']
        sampler = Sampler(
            name=experiment_id,
            **init_parameters.dim
        )
        sampler.setup(
            observations=observations,
            prior=Prior.generate_default_prior(
                var=prior_variance, **init_parameters.dim
            ),
            parameters=init_parameters,
        )
        evaluator = SamplerEvaluator(sampler,
                                     metric_functions=metric_functions,
                                     sample_functions=sample_functions,
                                     init_state=fit_state['evaluator_state'],
                                     )

    else:
        logger.info("Getting Init at {0}".format(path_to_init))
        init_parameters = joblib.load(path_to_init)
        sampler = Sampler(
            name=experiment_id,
            **init_parameters.dim
        )
        sampler.setup(
            observations=observations,
            prior=Prior.generate_default_prior(
                var=prior_variance, **init_parameters.dim
            ),
            parameters=init_parameters,
        )
        evaluator = SamplerEvaluator(sampler,
                                     metric_functions=metric_functions,
                                     sample_functions=sample_functions,
                                     )
        parameters_list = [
            dict(
                iteration=evaluator.iteration,
                elapsed_time=evaluator.elapsed_time,
                parameters=evaluator.sampler.parameters.copy()
            )
        ]

        # Save Init Figures
        logger.info("Saving Init Figures")
        process_checkpoint(
            evaluator=evaluator,
            data=data,
            parameters_list=parameters_list,
            experiment_id=experiment_id,
            path_to_out=path_to_out,
            path_to_fig=path_to_fig,
            checkpoint_num=evaluator.iteration,
        )

    # Sampler Funcs
    sampler_func_names, sampler_func_kwargs = get_model_sampler_step(
        model_type=model_type,
        inference_method=inference_method,
        steps_per_iteration=steps_per_iteration,
        **kwargs
    )
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    p_bar = tqdm(range(evaluator.iteration, max_num_iters),
                 file=tqdm_out, mininterval=60)

    last_checkpoint_time = time.time()
    last_eval_time = time.time() - eval_freq
    start_time = time.time()
    max_time_exceeded = False
    for step in p_bar:
        # Execute sampler_func_names
        if (time.time() - start_time > max_time):
            logger.info("Max Time Elapsed: {0} > {1}".format(
                time.time() - start_time, max_time))
            max_time_exceeded = True
        try:
            if (time.time() - last_eval_time > eval_freq) or \
                    (step == max_num_iters - 1) or max_time_exceeded:
                evaluator.evaluate_sampler_step(
                    sampler_func_names, sampler_func_kwargs, evaluate=True,
                )
                parameters_list.append(
                    dict(
                        iteration=evaluator.iteration,
                        elapsed_time=evaluator.elapsed_time,
                        parameters=evaluator.sampler.parameters.copy()
                    )
                )
                last_eval_time = time.time()
            else:
                evaluator.evaluate_sampler_step(
                    sampler_func_names, sampler_func_kwargs, evaluate=False,
                )
        except Exception as e:
            # Checkpoint On Error
            logger.error(f"Error during evaluation: {e}")
            process_checkpoint(
                evaluator=evaluator,
                data=data,
                parameters_list=parameters_list,
                experiment_id=experiment_id,
                path_to_out=path_to_out,
                path_to_fig=path_to_fig,
                checkpoint_num=evaluator.iteration,
            )
            fit_state = evaluator.get_state()
            logger.info("Saving Evaluator State to {0}".format(
                path_to_fit_state))
            joblib_write_to_file(
                dict(evaluator_state=fit_state,
                     parameters=evaluator.sampler.parameters,
                     parameters_list=parameters_list),
                path_to_fit_state)
            raise RuntimeError()

        # Check to Checkpoint Current Results
        if (step % checkpoint_num_iters == 0) or \
           (time.time() - last_checkpoint_time > checkpoint_time) or \
           (step == max_num_iters-1) or max_time_exceeded:
            process_checkpoint(
                evaluator=evaluator,
                data=data,
                parameters_list=parameters_list,
                experiment_id=experiment_id,
                path_to_out=path_to_out,
                path_to_fig=path_to_fig,
                checkpoint_num=evaluator.iteration,
            )
            fit_state = evaluator.get_state()
            logger.info("Saving Evaluator State to {0}".format(
                path_to_fit_state))
            joblib_write_to_file(
                dict(evaluator_state=fit_state,
                     parameters=evaluator.sampler.parameters,
                     parameters_list=parameters_list),
                path_to_fit_state)

            # Reset Checkpoint Clock
            last_checkpoint_time = time.time()

        if max_time_exceeded:
            break

    return evaluator

# Evaluate Module


def do_eval(target,
            experiment_name, experiment_id,
            experiment_folder,
            model_type, prior_variance,
            max_eval_iterations, max_eval_time,
            checkpoint_num_iters, checkpoint_time,
            **kwargs):

    logger.info("Beginning Evaluation of {0} id:{1} on {2}".format(
        experiment_name, experiment_id, target,
    ))

    Sampler, Prior = get_model_sampler_prior(model_type)

    # Ensure we have an absolute path for the experiment folder
    if not os.path.isabs(experiment_folder):
        experiment_folder = os.path.abspath(experiment_folder)
    logger.info("Using absolute experiment folder path: {}".format(
        experiment_folder))

    # Paths - Normalize all paths to ensure consistent directory separators
    experiment_folder = normalize_path(experiment_folder)
    path_to_parameters_list = normalize_path(os.path.join(experiment_folder, "out", "fit",
                                                          "{0}_parameters.p".format(experiment_id)))
    path_to_out = normalize_path(os.path.join(experiment_folder, "out",
                                              "eval{0}".format(target)))
    path_to_fig = normalize_path(os.path.join(experiment_folder, "fig",
                                              "eval{0}".format(target), "{0:0>4}".format(experiment_id)))
    path_to_scratch = normalize_path(
        os.path.join(experiment_folder, 'scratch'))
    path_to_eval_state = normalize_path(os.path.join(path_to_scratch,
                                                     "eval{1}_{0:0>4}_state.p".format(experiment_id, target)))

    # Log the paths for debugging
    logger.info("Parameters list path: {0}".format(path_to_parameters_list))
    logger.info("Output path: {0}".format(path_to_out))
    logger.info("Figure path: {0}".format(path_to_fig))
    logger.info("Scratch path: {0}".format(path_to_scratch))
    logger.info("Eval state path: {0}".format(path_to_eval_state))

    # Check if the parameters file exists
    if not os.path.isfile(path_to_parameters_list):
        logger.error("Parameters file not found: {}".format(
            path_to_parameters_list))
        # Try alternate locations
        alt_path = normalize_path(os.path.join("lgssm_demo", "out", "fit",
                                               "{0}_parameters.p".format(experiment_id)))
        logger.info("Trying alternate path: {}".format(alt_path))
        if os.path.isfile(alt_path):
            path_to_parameters_list = alt_path
            logger.info("Found parameters at alternate path")
        else:
            # Try to find the file
            logger.info("Searching for parameters file...")
            for root, _, files in os.walk(os.path.dirname(experiment_folder)):
                for file in files:
                    if file == "{0}_parameters.p".format(experiment_id):
                        path_to_parameters_list = normalize_path(
                            os.path.join(root, file))
                        logger.info("Found parameters at: {}".format(
                            path_to_parameters_list))
                        break

    safe_make_path(path_to_out)
    safe_make_path(path_to_fig)
    safe_make_path(path_to_scratch)

    # Get Data
    if target in ["train", "half_avg_train"]:
        path_to_data = kwargs['path_to_data']
        logger.info("Getting Data at {0}".format(path_to_data))
        data = joblib.load(path_to_data)
    elif target in ["test", "half_avg_test"]:
        path_to_data = kwargs['path_to_test_data']
        logger.info("Getting Data at {0}".format(path_to_data))
        data = joblib.load(path_to_data)
    else:
        raise ValueError("Invalid target {0}".format(target))

    # Setup Sampler
    logger.info("Setting up Sampler")
    path_to_init = kwargs['path_to_init']
    init_parameters = joblib.load(path_to_init)
    sampler = Sampler(
        name=experiment_id,
        **init_parameters.dim
    )
    observations = data['observations']
    sampler.setup(
        observations=observations,
        prior=Prior.generate_default_prior(
            var=prior_variance,
            **init_parameters.dim
        ),
    )

    # Set Metric + Sample Functions for Evaluator
    parameter_names = ['A', 'C', 'Q', 'R']
    sample_functions = [sample_function_parameters(parameter_names)]
    metric_functions = [noisy_logjoint_loglike_metric()]
    if 'parameters' in data.keys():
        metric_functions += [
            metric_function_parameters(
                parameter_names=parameter_names,
                target_values=[getattr(data['parameters'], parameter_name)
                               for parameter_name in parameter_names],
                metric_names=['logmse' for _ in parameter_names],
            )
        ]
    if 'latent_vars' in data.keys():
        metric_functions += [metric_compare_x(true_x=data['latent_vars'])]

    # Get parameters_list
    logger.info("Getting Params from {0}".format(path_to_parameters_list))
    parameters_list = joblib.load(path_to_parameters_list)
    if target in ["half_avg_train", "half_avg_test"]:
        logger.info("Calculating Running Average of Parameters")
        parameters_list['parameters'] = \
            half_average_parameters_list(parameters_list['parameters'])

    # Setup Evaluator
    logger.info("Setting up Evaluator")
    # Check if existing evaluator state exists
    if os.path.isfile(path_to_eval_state):
        logger.info("Continuing Evaluation from {0}".format(
            path_to_eval_state))
        eval_state = joblib.load(path_to_eval_state)
        evaluator = OfflineEvaluator(sampler,
                                     parameters_list=parameters_list,
                                     metric_functions=metric_functions,
                                     sample_functions=sample_functions,
                                     init_state=eval_state,
                                     )

    else:
        logger.info("Initializing Evaluation from scratch")
        evaluator = OfflineEvaluator(sampler,
                                     parameters_list=parameters_list,
                                     metric_functions=metric_functions,
                                     sample_functions=sample_functions,
                                     )
        process_checkpoint(
            evaluator=evaluator,
            data=data,
            experiment_id=experiment_id,
            path_to_out=path_to_out,
            path_to_fig=path_to_fig,
        )

    # Evaluation
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    logger.info("Found {0} parameters to eval".format(evaluator.num_to_eval()))
    max_iterations = min([max_eval_iterations, evaluator.num_to_eval()])
    p_bar = tqdm(range(max_iterations),  file=tqdm_out, mininterval=60)

    last_checkpoint_time = time.time() - checkpoint_time
    start_time = time.time()
    max_time_exceeded = False
    for p_iter in p_bar:
        if (time.time() - start_time > max_eval_time):
            logger.info("Max Time Elapsed: {0} > {1}".format(
                time.time() - start_time, max_eval_time))
            max_time_exceeded = True

        # Offline Evaluation
        evaluator.evaluate(num_to_eval=1)

        if ((time.time()-last_checkpoint_time) > checkpoint_time) or \
                (p_iter == max_iterations-1) or max_time_exceeded:
            process_checkpoint(
                evaluator=evaluator,
                data=data,
                experiment_id=experiment_id,
                path_to_out=path_to_out,
                path_to_fig=path_to_fig,
            )
            eval_state = evaluator.get_state()
            logger.info("Saving Evaluator State to {0}".format(
                path_to_eval_state))
            joblib_write_to_file(eval_state, path_to_eval_state)

            # Reset Checkpoint Clock
            last_checkpoint_time = time.time()

        if max_time_exceeded:
            break

    return evaluator

# Combine dfs from individual experiments


def do_process_out(experiment_folder):
    """ Process Output
        Aggregate files of form .../out/../{id}_{**}.csv

    """
    experiment_folder = normalize_path(experiment_folder)
    path_to_out = normalize_path(os.path.join(experiment_folder, 'out'))
    path_to_options = normalize_path(os.path.join(
        experiment_folder, 'in', 'options.csv'))

    path_to_processed = normalize_path(
        os.path.join(experiment_folder, "processed"))
    safe_make_path(path_to_processed)

    subfolders = os.listdir(path_to_out)
    # Copy Options to processed
    logger.info("Copying Options")
    options_df = pd.read_csv(path_to_options, index_col=False)
    pandas_write_df_to_csv(options_df,
                           filename=os.path.join(
                               path_to_processed, "options.csv"),
                           index=False)

    # Try to Aggregate Data [evaltrain+evaltest, fit_metrics[time], options]
    aggregated_columns = [
        'iteration', 'metric', 'value', 'variable',
        'eval_set', 'time', 'iteration_time', 'experiment_id',
    ]
    evaltargets = ['evaltrain', 'evalhalf_avg_train',
                   'evaltest', 'evalhalf_avg_test']
    if ('fit' in subfolders) and (len(set(subfolders).intersection(
            set(evaltargets))) > 0):
        path_to_aggregated_df = os.path.join(
            path_to_processed, "aggregated.csv")
        logger.info("Aggregating Data to {0}".format(path_to_aggregated_df))

        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        p_bar = tqdm(list(enumerate(options_df['experiment_id'].unique())),
                     file=tqdm_out, mininterval=60)

        new_csv_flag = True
        for ii, experiment_id in p_bar:
            eval_df = pd.DataFrame()
            for evaltarget in evaltargets:
                # LOAD EVAL TARGET FILE
                if evaltarget in subfolders:
                    eval_target_file = os.path.join(path_to_out, evaltarget,
                                                    '{0}_metrics.csv'.format(
                                                        experiment_id),
                                                    )
                    if not is_valid_file(eval_target_file):
                        continue
                    eval_target_df = pd.read_csv(
                        eval_target_file, index_col=False,
                    ).assign(eval_set=evaltarget)
                    eval_df = pd.concat([eval_df, eval_target_df],
                                        ignore_index=True)

            # LOAD FIT FILE
            fit_file = os.path.join(path_to_out, 'fit',
                                    '{0}_metrics.csv'.format(experiment_id),
                                    )
            if not is_valid_file(fit_file):
                continue

            fit_df = pd.read_csv(fit_file, index_col=False)
            fit_df = fit_df[fit_df['iteration'].isin(eval_df['iteration'])]
            iteration_time = fit_df.query("metric == 'time'")[
                ['iteration', 'value']].rename(
                columns={'value': 'iteration_time'})
            run_time = fit_df.query("metric == 'runtime'")[
                ['iteration', 'value']].rename(
                columns={'value': 'time'})

            df = pd.merge(eval_df, iteration_time,
                          how='left', on=['iteration'])
            df = pd.merge(df, run_time, how='left', on=['iteration'])
            df = df.sort_values('iteration').assign(
                experiment_id=experiment_id)

            if new_csv_flag:
                df[aggregated_columns].to_csv(path_to_aggregated_df,
                                              index=False)
                new_csv_flag = False
            else:
                df.reindex(columns=aggregated_columns).to_csv(
                    path_to_aggregated_df, mode='a', header=False,
                    index=False)
        logger.info("Done Aggregating Data: {0}".format(path_to_aggregated_df))

    # Also concat out folder csvs
    for subfolder in subfolders:
        # Only Process Folders
        path_to_subfolder = os.path.join(path_to_out, subfolder)
        if not os.path.isdir(path_to_subfolder):
            logger.info("Ignoring file {0}".format(subfolder))
            continue

        logger.info("Combining Data in Folder {0}".format(path_to_subfolder))
        filenames = os.listdir(path_to_subfolder)

        # Combine Metrics
        metric_filenames = [name for name in filenames
                            if name.endswith("metrics.csv")]
        path_to_metric_df = os.path.join(path_to_processed,
                                         "{0}_metrics.csv".format(subfolder))
        logger.info("Aggregating Data to {0}".format(path_to_metric_df))

        # Concat by appending to one large csv
        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        p_bar = tqdm(list(enumerate(metric_filenames)), file=tqdm_out,
                     mininterval=60)

        new_csv_flag = True
        for ii, name in p_bar:
            file_name = os.path.join(path_to_subfolder, name)
            if not is_valid_file(file_name):
                continue
            metric_df = pd.read_csv(file_name, index_col=False)
            metric_df['experiment_id'] = name.split("_")[0]
            if new_csv_flag:
                metric_df.to_csv(path_to_metric_df, index=False)
                metric_df_columns = list(metric_df.columns.values)
                new_csv_flag = False
            else:
                metric_df.reindex(columns=metric_df_columns).to_csv(
                    path_to_metric_df, mode='a', header=False, index=False)
        logger.info("Metric Data Aggregated to {0}".format(path_to_metric_df))

    return


# Make Quick Plots
def do_make_plots(experiment_folder):
    """ Make quick plots based on aggregated.csv output of `do_process_out` """
    experiment_folder = normalize_path(experiment_folder)
    path_to_processed = normalize_path(
        os.path.join(experiment_folder, 'processed'))
    path_to_fig = normalize_path(os.path.join(
        experiment_folder, 'fig', 'processed'))
    safe_make_path(path_to_fig)

    logger.info("Loading Data")
    aggregated_df = pd.read_csv(
        os.path.join(path_to_processed, 'aggregated.csv'))
    options_df = pd.read_csv(
        os.path.join(path_to_processed, 'options.csv'))

    evaltargets = aggregated_df['eval_set'].unique()
    logger.info("Making Plots for {0}".format(evaltargets))
    for evaltarget in evaltargets:
        logger.info("Processing Data for {0}".format(evaltarget))
        sub_df = pd.merge(
            aggregated_df[aggregated_df['eval_set'] == evaltarget],
            options_df[['method_name', 'experiment_id']],
            on='experiment_id',
        )
        sub_df['variable_metric'] = sub_df['variable'] + '_' + sub_df['metric']

        logger.info("Plotting metrics vs time for {0}".format(evaltarget))
        plt.close('all')
        g = sns.relplot(x='time', y='value', hue='method_name', kind='line',
                        col='variable_metric', col_wrap=3,
                        estimator=None, units='experiment_id',
                        data=sub_df,
                        facet_kws=dict(sharey=False),
                        )
        g.fig.set_size_inches(12, 10)
        g.savefig(os.path.join(path_to_fig,
                               '{0}_metric_vs_time.png'.format(evaltarget)))

        logger.info("Plotting metrics vs iteration for {0}".format(evaltarget))
        plt.close('all')
        g = sns.relplot(x='iteration', y='value', hue='method_name',
                        kind='line', col='variable_metric', col_wrap=3,
                        estimator=None, units='experiment_id',
                        data=sub_df,
                        facet_kws=dict(sharey=False),
                        )
        g.fig.set_size_inches(12, 10)
        g.savefig(os.path.join(path_to_fig,
                               '{0}_metric_vs_iteration.png'.format(evaltarget)))

        # After Burnin
        if sub_df.query('iteration > 100').shape[0] > 0:
            logger.info(
                "Plotting metrics vs time after burnin for {0}".format(evaltarget))
            plt.close('all')
            g = sns.relplot(x='time', y='value', hue='method_name', kind='line',
                            col='variable_metric', col_wrap=3,
                            estimator=None, units='experiment_id',
                            data=sub_df.query('iteration > 100'),
                            facet_kws=dict(sharey=False),
                            )
            g.fig.set_size_inches(12, 10)
            g.savefig(os.path.join(path_to_fig,
                                   '{0}_metric_vs_time_burnin.png'.format(evaltarget)))

            logger.info(
                "Plotting metrics vs iteration for {0}".format(evaltarget))
            plt.close('all')
            g = sns.relplot(x='iteration', y='value', hue='method_name',
                            kind='line', col='variable_metric', col_wrap=3,
                            estimator=None, units='experiment_id',
                            data=sub_df.query('iteration > 100'),
                            facet_kws=dict(sharey=False),
                            )
            g.fig.set_size_inches(12, 10)
            g.savefig(os.path.join(path_to_fig,
                                   '{0}_metric_vs_iteration_burnin.png'.format(evaltarget)))

    return

# Evaluate Parameter Sample Quality


def do_eval_ksd(
        experiment_name, experiment_id,
        experiment_folder,
        model_type, prior_variance,
        max_eval_iterations, max_eval_time,
        checkpoint_num_iters, checkpoint_time,
        ksd_burnin=0.33, ksd_subsequence_length=1000, ksd_buffer_length=10,
        **kwargs):
    """ Evaluate the Kernelized Stein Divergence

    Pseudocode:
        Load Train Data + Setup Sampler
        Load Parameter Trace for Experiment Id (apply burnin)
        For each parameter, calculate the gradient of the logjoint
           (if using noisy gradients, take average over multiple replications)
        Compute KSD at each checkpoint
        Checkpoints results to out/eval_ksd for each experiment_id
    """
    from sgmcmc_ssm.trace_metric_functions import compute_KSD
    GRAD_DIM = 3
    GRAD_VARIABLES = ['A', 'Q', 'R']

    logger.info("Beginning KSD Evaluation of {0} id:{1}".format(
        experiment_name, experiment_id,
    ))
    # Paths
    experiment_folder = normalize_path(experiment_folder)
    path_to_parameters_list = normalize_path(os.path.join(experiment_folder, "out", "fit",
                                                          "{0}_parameters.p".format(experiment_id)))
    path_to_out = normalize_path(os.path.join(experiment_folder, "/out",
                                              "trace_eval_ksd"))
    path_to_scratch = normalize_path(
        os.path.join(experiment_folder, 'scratch'))
    path_to_checkpoint_state = normalize_path(os.path.join(path_to_scratch,
                                                           "trace_eval_ksd_{0:0>4}_state.p".format(experiment_id)))

    safe_make_path(path_to_out)
    safe_make_path(path_to_scratch)

    # Load Train Data + Setup Sampler
    Sampler, Prior = get_model_sampler_prior(model_type)
    path_to_data = kwargs['path_to_data']
    logger.info("Getting Data at {0}".format(path_to_data))
    data = joblib.load(path_to_data)

    logger.info("Setting up Sampler")
    path_to_init = kwargs['path_to_init']
    init_parameters = joblib.load(path_to_init)
    sampler = Sampler(
        name=experiment_id,
        **init_parameters.dim
    )
    observations = data['observations']
    sampler.setup(
        observations=observations,
        prior=Prior.generate_default_prior(
            var=prior_variance,
            **init_parameters.dim
        ),
    )

    if not os.path.isfile(path_to_checkpoint_state):
        # Load parameter_list
        logger.info("Getting Params from {0}".format(path_to_parameters_list))
        parameters_list = joblib.load(path_to_parameters_list).copy()
        # Apply Burnin
        parameters_list = parameters_list.iloc[int(
            parameters_list.shape[0]*ksd_burnin):]
        parameters_list['num_ksd_eval'] = 0.0
        parameters_list['grad'] = [
            np.zeros(GRAD_DIM) for _ in range(parameters_list.shape[0])]
        # Add grad_name column with default value 'A'
        parameters_list['grad_name'] = 'A'
        metrics_df = pd.DataFrame()
        cur_param_index = 0
        logger.info("Calculating KSD on {0} parameters".format(
            parameters_list.shape[0]))
    else:
        # Load metrics_df + parameter_list from checkpoint
        logger.info("Loading parameters from previous checkpoint")
        checkpoint_state = joblib.load(path_to_checkpoint_state)
        parameters_list = checkpoint_state['parameters_list']
        metrics_df = checkpoint_state['metrics_df']
        cur_param_index = checkpoint_state['cur_param_index']

        # Ensure grad_name column exists
        if 'grad_name' not in parameters_list.columns:
            parameters_list['grad_name'] = 'A'

        logger.info("Found {0} parameters with at least {1} evals".format(
            parameters_list.shape[0], parameters_list['num_ksd_eval'].min()))

        # Terminate after 1 pass if exact KSD
        if (ksd_subsequence_length == -1) or \
                (ksd_subsequence_length >= data['observations'].shape[0]):
            if (cur_param_index == 0) and \
                    (parameters_list['num_ksd_eval'].min() >= 1):
                logger.info("Already computed exact KSD")
                return metrics_df

    max_iterations = max_eval_iterations*parameters_list.shape[0]
    start_time = time.time()
    max_time_exceeded = False
    last_checkpoint_time = time.time()
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    p_bar = tqdm(range(max_iterations), file=tqdm_out, mininterval=60)
    for ii in p_bar:
        if (time.time() - start_time > max_eval_time):
            logger.info("Max Time Elapsed: {0} > {1}".format(
                time.time() - start_time, max_eval_time))
            max_time_exceeded = True

        parameters = parameters_list['parameters'].iloc[cur_param_index]
        sampler.parameters = parameters

        # Get the gradient from the sampler
        noisy_grad = sampler.noisy_gradient(
            subsequence_length=ksd_subsequence_length,
            buffer_length=ksd_buffer_length,
            is_scaled=False)

        # Debug the raw gradient
        logger.info("Raw noisy gradient type: {}, value: {}".format(
            type(noisy_grad), noisy_grad))

        # Extract numeric gradient if it's a dictionary or contains a dictionary
        if isinstance(noisy_grad, dict):
            if 'A' in noisy_grad:
                grad_name = 'A'
                gradient_value = noisy_grad['A']
            elif 'LQinv_vec' in noisy_grad:  # Try a different key
                grad_name = 'A'  # Default to A
                # Use this as an approximation
                gradient_value = noisy_grad['LQinv_vec']
            else:
                logger.warning(
                    "Unknown gradient dictionary structure: {}".format(noisy_grad))
                grad_name = 'A'
                gradient_value = 0.0
        elif isinstance(noisy_grad, np.ndarray) and noisy_grad.size > 0 and isinstance(noisy_grad[0], dict):
            # The gradient is an array with a dictionary as first element
            dict_value = noisy_grad[0]
            logger.info("Dictionary in gradient: {}".format(dict_value))
            if 'A' in dict_value:
                grad_name = 'A'
                # Extract A value - could be array or scalar
                a_value = dict_value['A']
                if hasattr(a_value, 'item') and a_value.size == 1:
                    gradient_value = a_value.item()
                elif hasattr(a_value, 'flatten'):
                    gradient_value = a_value.flatten(
                    )[0] if a_value.size > 0 else 0.0
                else:
                    gradient_value = a_value
                logger.info("Extracted A value: {}".format(gradient_value))
            else:
                grad_name = 'A'
                gradient_value = 0.0
                logger.warning(
                    "Dictionary doesn't have expected keys: {}".format(dict_value))
        else:
            grad_name = parameters_list['grad_name'].iloc[cur_param_index]
            gradient_value = noisy_grad

        # Now convert the gradient using the numeric value
        grad = convert_gradient(gradient=gradient_value, grad_name=grad_name)

        logger.info(
            "Processed gradient type: {}, value: {}".format(type(grad), grad))

        index = parameters_list.index[cur_param_index]
        current_grad = parameters_list.at[index, 'grad']
        logger.info("Current grad type: {}, value: {}".format(
            type(current_grad), current_grad))

        # Add gradients safely - reinitialize if necessary
        if not isinstance(current_grad, np.ndarray):
            logger.info("Reinitializing gradient at index {}".format(index))
            parameters_list.at[index, 'grad'] = np.zeros(GRAD_DIM)
            current_grad = parameters_list.at[index, 'grad']

        # Now add the gradients
        parameters_list.at[index, 'grad'] = current_grad + grad
        parameters_list.at[index, 'num_ksd_eval'] += 1.0

        # Update parameter index for next loop
        cur_param_index += 1
        if cur_param_index == parameters_list.shape[0]:
            logger.info("Completed {0} passes over all parameters".format(
                parameters_list['num_ksd_eval'].min()))
            cur_param_index = 0

        # Checkpoint Results
        if ((time.time() - last_checkpoint_time > checkpoint_time) or
            (cur_param_index == 0) or (ii+1 == max_eval_iterations) or
                max_time_exceeded):
            # Compute KSD
            sub_list = parameters_list[parameters_list['num_ksd_eval'] > 0]
            param_list = sub_list['parameters']
            grad_list = sub_list['grad'] / sub_list['num_ksd_eval']

            result_dict = compute_KSD(
                param_list=param_list.tolist(), grad_list=grad_list.tolist(),
                variables=GRAD_VARIABLES,
                max_block_size=512,  # Block Size for computing kernel
            )

            new_metric_df = pd.DataFrame([
                dict(metric='ksd', variable=key, value=value,
                     num_samples=cur_param_index-1,
                     num_evals=parameters_list['num_ksd_eval'].min(),
                     ) for key, value in result_dict.items()
            ])
            metrics_df = pd.concat([metrics_df, new_metric_df],
                                   ignore_index=True)

            # Save Metrics DF to CSV
            path_to_metrics_file = os.path.join(path_to_out,
                                                "{0}_metrics.csv".format(experiment_id))
            logger.info("Saving KSD metrics to {0}".format(
                path_to_metrics_file))
            pandas_write_df_to_csv(
                metrics_df, path_to_metrics_file, index=False)

            # Checkpoint State
            logger.info("Saving checkpoint to {0}".format(
                path_to_checkpoint_state))
            joblib_write_to_file(dict(
                parameters_list=parameters_list,
                metrics_df=metrics_df,
                cur_param_index=cur_param_index,
            ), path_to_checkpoint_state)

            # Reset Checkpoint Clock
            last_checkpoint_time = time.time()

            # Terminate after 1 pass if exact KSD
            if (ksd_subsequence_length == -1) or \
                    (ksd_subsequence_length >= data['observations'].shape[0]):
                if cur_param_index == 0:
                    break

            # Terminate if max_time_exceeded
            if max_time_exceeded:
                break

    return metrics_df


def do_eval_ks_test(
        experiment_name, experiment_id,
        experiment_folder,
        model_type, prior_variance,
        max_eval_iterations, max_eval_time,
        checkpoint_num_iters, checkpoint_time,
        kstest_burnin=0.33, kstest_variables=None,
        path_to_reference_parameter_list=None,
        **kwargs):
    """ Evaluate KS Test on Markov Chain """
    # Setup Paths
    experiment_id_str = str(experiment_id).zfill(3)
    path_to_out = normalize_path(os.path.join(
        experiment_folder, "out", "trace_eval"))
    path_to_fig = normalize_path(os.path.join(
        experiment_folder, "fig", "trace_eval"))
    logger.info("Saving Results to {0}".format(path_to_out))
    logger.info("Saving Figures to {0}".format(path_to_fig))
    safe_make_path(path_to_out)
    safe_make_path(path_to_fig)

    # Get Reference Parameter List
    if path_to_reference_parameter_list is None:
        # Read Reference Parameter List from Setup
        path_to_input = normalize_path(os.path.join(
            experiment_folder, "in"))
        path_to_options = os.path.join(path_to_input, "options.p")
        if not is_valid_file(path_to_options):
            return
        arg_list = joblib.load(path_to_options)
        arg_dict = arg_list[experiment_id]
        reference_data_path = arg_dict.get("path_to_data")
        reference_data = joblib.load(reference_data_path)
        reference_parameters = {'parameters': [reference_data['parameters']]}
    else:
        reference_parameters = joblib.load(path_to_reference_parameter_list)

    if kstest_variables is None:
        if model_type == "LGSSM":
            kstest_variables = ['A', 'C', 'Q', 'R']
        else:
            kstest_variables = ['phi', 'sigma', 'tau']

    logger.info("Beginning KS Test Evaluation of {0} id:{1}".format(
        experiment_name, experiment_id,
    ))
    # Paths
    experiment_folder = normalize_path(experiment_folder)
    path_to_parameters_list = normalize_path(os.path.join(experiment_folder, "out", "fit",
                                                          "{0}_parameters.p".format(experiment_id)))
    path_to_out = normalize_path(os.path.join(experiment_folder, "out",
                                              "trace_eval_kstest"))
    path_to_fig = normalize_path(os.path.join(experiment_folder, "fig",
                                              "trace_eval_kstest"))
    safe_make_path(path_to_out)
    safe_make_path(path_to_fig)

    # Load Experiment ID Parameter Trace
    logger.info("Getting Params from {0}".format(path_to_parameters_list))
    parameters_list = joblib.load(path_to_parameters_list)
    # Apply Burnin
    parameters_list = parameters_list.iloc[int(parameters_list.shape[0]*0.33):]
    parameters_list = parameters_list[['parameters']]

    # Calculate KSTest for each variable
    metrics_df = pd.DataFrame()
    cur_param_index = 0
    logger.info("Calculating KS-Test on {0} parameters".format(
        parameters_list.shape[0]))

    results = []
    plt.close('all')
    fig, axes = plt.subplots(1, len(kstest_variables), sharey=False)
    for ii, variable in enumerate(kstest_variables):
        data_ref = np.array([getattr(param, variable)
                             for param in reference_parameters['parameters']]).flatten()
        data_samp = np.array([getattr(param, variable)
                              for param in parameters_list['parameters']]).flatten()
        statistic, pvalue = ks_2samp(data_samp, data_ref)
        results.append(dict(metric='kstest', variable=variable,
                            value=statistic))
        results.append(dict(metric='kstest_pvalue', variable=variable,
                            value=pvalue))

        sns.distplot(data_ref, ax=axes[ii], label='ref')
        sns.distplot(data_samp, ax=axes[ii], label='samp')
        if pvalue < 0.05:
            axes[ii].set_title('{0}\n KS-value: {1:1.2e} ({2:1.2e}*)'.format(
                variable, statistic, pvalue))
        else:
            axes[ii].set_title('{0}\n KS-value: {1:1.2e} ({2:1.2e})'.format(
                variable, statistic, pvalue))
    axes[-1].legend()
    fig.set_size_inches(4*len(kstest_variables), 7)
    fig.savefig(os.path.join(path_to_fig, "{0}_trace_density.png".format(
        experiment_id)))

    results.append(dict(metric='num_samples', variable="trace",
                        value=parameters_list.shape[0]))
    metrics_df = pd.DataFrame(results)

    # Save Metrics DF to CSV
    path_to_metrics_file = os.path.join(path_to_out,
                                        "{0}_metrics.csv".format(experiment_id))
    logger.info("Metrics:\n{0}".format(metrics_df))
    logger.info("Saving KSTest metrics to {0}".format(path_to_metrics_file))
    pandas_write_df_to_csv(metrics_df, path_to_metrics_file, index=False)

    return metrics_df


###############################################################################
# Experiment Specific Functions
###############################################################################
def setup_train_test_data(experiment_folder, experiment_name, T, T_test,
                          parameter_list, data_reps, init_methods, **kwargs):
    """ Setup Synthetic Data """
    # Setup Input Folder
    path_to_input = normalize_path(os.path.join(experiment_folder, "in"))
    safe_make_path(path_to_input)

    # Generate Training + Test Data
    logger.info("Generating Training Data + Inits")
    input_args = []

    # Create + Save Test Data (shared among training sets)
    for param_num, (param_name, parameters) in enumerate(parameter_list.items()):
        test_data = generate_lgssm_data(T=T_test, parameters=parameters)
        test_data_name = "test_data"
        path_to_test_data = os.path.join(path_to_input,
                                         "{0}.p".format(test_data_name))
        joblib.dump(test_data, path_to_test_data)

        for data_rep in range(data_reps):
            # Create + Save Training Data
            train_data = generate_lgssm_data(T=T, parameters=parameters)
            data_name = "train_data_{0}".format(data_rep+data_reps*param_num)
            path_to_data = os.path.join(path_to_input,
                                        "{0}.p".format(data_name))
            joblib.dump(train_data, path_to_data)

            # Generate Inits
            for init_num, init_method in enumerate(init_methods):
                logger.info("Generating Init {0} of {1}".format(
                    init_num, len(init_methods)))
                # Create + Save Init
                path_to_init = os.path.join(path_to_input,
                                            "{0}_init_{1}.p".format(data_name, init_num))
                setup_init(
                    data=train_data,
                    init_method=init_method,
                    path_to_init=path_to_init,
                )
                input_args.append({
                    'experiment_name': experiment_name,
                    'path_to_data': path_to_data,
                    'path_to_test_data': path_to_test_data,
                    'path_to_init': path_to_init,
                    'param_name': param_name,
                    'init_method': init_method,
                })
    return input_args


def setup_init(data, init_method, path_to_init, n=1, m=1):
    """ Setup Init Parameters for data """
    if init_method == "prior":
        prior = LGSSMPrior.generate_default_prior(n=n, m=m)
        sampler = LGSSMSampler(n=n, m=m)
        sampler.setup(observations=data['observations'],
                      prior=prior)
        sampler.project_parameters()
        init_parameters = sampler.parameters
    elif init_method == "truth":
        init_parameters = data['parameters']
    else:
        raise ValueError("Unrecognized init_method")
    joblib.dump(init_parameters, path_to_init)
    return init_parameters


def setup_options(experiment_folder, arg_list):
    # Create Options csv in <experiment_folder>/in
    path_to_input = normalize_path(os.path.join(experiment_folder, "in"))
    safe_make_path(path_to_input)

    # Sort Arg List by Data x Init Trial
    arg_list = sorted(arg_list,
                      key=lambda k: (k['path_to_data'], k['path_to_init']))

    # Assign Experiment ID + Experiment Folder Location
    for ii, custom_dict in enumerate(arg_list):
        # Set Defaults
        arg_dict = DEFAULT_OPTIONS.copy()
        arg_dict.update(custom_dict)
        arg_dict["experiment_id"] = ii
        arg_dict["experiment_folder"] = experiment_folder
        arg_list[ii] = arg_dict

    path_to_arg_list = os.path.join(path_to_input, "options.p")
    logger.info("Saving arg_list as {0}".format(path_to_arg_list))
    joblib.dump(arg_list, path_to_arg_list)

    options_df = pd.DataFrame(arg_list)
    path_to_options_file = os.path.join(path_to_input, "options.csv")
    logger.info("Also saving as csv at {0}".format(path_to_options_file))
    options_df.to_csv(path_to_options_file, index=False)

    return options_df


def get_model_sampler_prior(model_type):
    if model_type == "LGSSM":
        Sampler = LGSSMSampler
        Prior = LGSSMPrior
    else:
        raise NotImplementedError()
    return Sampler, Prior


def get_model_sampler_step(
        model_type, inference_method, steps_per_iteration,
        epsilon, minibatch_size, subsequence_length, buffer_length,
        **kwargs):
    """ Returns sampler_func_names + sampler_func_kwargs for SamplerEvaluator"""
    step_kwargs = dict(
        epsilon=epsilon,
        minibatch_size=minibatch_size,
        subsequence_length=subsequence_length,
        buffer_length=buffer_length,
        kind=kwargs.get("kind", "marginal"),
        num_samples=kwargs.get("num_samples", None),
        **kwargs.get("pf_kwargs", {})
    )
    if inference_method in ['SGRD', 'SGRLD']:
        if 'preconditioner' not in step_kwargs.keys():
            step_kwargs['preconditioner'] = LGSSMPreconditioner()

    if inference_method == 'SGD':
        sampler_func_names = ['step_sgd', 'project_parameters']
        sampler_func_kwargs = [step_kwargs, {}]
    elif inference_method == 'ADAGRAD':
        sampler_func_names = ['step_adagrad', 'project_parameters']
        sampler_func_kwargs = [step_kwargs, {}]
    elif inference_method == 'SGRD':
        sampler_func_names = ['step_sgd', 'project_parameters']
        sampler_func_kwargs = [step_kwargs, {}]
    elif inference_method == 'SGLD':
        sampler_func_names = ['sample_sgld', 'project_parameters']
        sampler_func_kwargs = [step_kwargs, {}]
    elif inference_method == 'SGRLD':
        sampler_func_names = ['sample_sgrld', 'project_parameters']
        sampler_func_kwargs = [step_kwargs, {}]
    elif inference_method == 'Gibbs':
        sampler_func_names = ["sample_gibbs", "project_parameters"]
        sampler_func_kwargs = [{}, {}]

    sampler_func_names = sampler_func_names * steps_per_iteration
    sampler_func_kwargs = sampler_func_kwargs * steps_per_iteration
    return sampler_func_names, sampler_func_kwargs

###############################################################################
# Helper / Utility Functions


def dict_product(*args):
    # Combine a list of dictionary lists
    from itertools import product
    return [{k: v for d in L for k, v in d.items()} for L in product(*args)]


def update_dict(ldict, rdict):
    """ Update ldict with key, value pairs from rdict """
    updated_dict = ldict.copy()
    updated_dict.update(rdict)
    return updated_dict


def is_valid_file(filename):
    # Check filename exists + is not empty
    if not os.path.isfile(filename):
        logging.info("Missing File {0}".format(filename))
        return False
    elif os.path.getsize(filename) <= 1:
        # File is currently being written
        logging.info("Pausing for 5.0 sec for {0}".format(filename))
        time.sleep(5.0)
        if os.path.getsize(filename) <= 1:
            logging.info("== EMPTY File {0} ==".format(filename))
            return False
    else:
        return True


def process_checkpoint(evaluator, data, experiment_id,
                       path_to_out, path_to_fig,
                       checkpoint_num=0, parameters_list=None,
                       **kwargs):
    """ Save Checkpoint """
    # Normalize paths
    path_to_out = normalize_path(path_to_out)
    path_to_fig = normalize_path(path_to_fig)

    # Make sure paths exist
    safe_make_path(path_to_out)
    safe_make_path(path_to_fig)

    # Save Metrics
    path_to_metrics_file = normalize_path(os.path.join(path_to_out,
                                                       "{0}_metrics.csv".format(experiment_id)))
    logger.info("Saving Metrics to {0}".format(path_to_metrics_file))
    pandas_write_df_to_csv(df=evaluator.get_metrics(),
                           filename=path_to_metrics_file, index=False)

    if parameters_list is not None:
        path_to_parameters_list = normalize_path(os.path.join(path_to_out,
                                                              "{0}_parameters.p".format(experiment_id)))
        logger.info("Saving Parameters List to {0}".format(
            path_to_parameters_list))
        joblib_write_to_file(pd.DataFrame(parameters_list),
                             filename=path_to_parameters_list)

    if len(evaluator.metric_functions) > 0 and evaluator.metrics.shape[0] > 0:
        path_to_metrics_plot = os.path.join(path_to_fig, "metrics.png")
        logger.info("Plotting Metrics to {0}".format(path_to_metrics_plot))
        plt.close('all')
        g = plot_metrics(evaluator)
        g.fig.set_size_inches(12, 10)
        g.savefig(path_to_metrics_plot)

        if len(evaluator.metrics['iteration'].unique()) > 10:
            path_to_zoom_metrics_plot = \
                os.path.join(path_to_fig, "metrics_zoom.png")
            logger.info("Plotting Zoom Metrics to {0}".format(
                path_to_zoom_metrics_plot))
            plt.close('all')
            g = plot_metrics(evaluator, full_trace=False)
            g.fig.set_size_inches(12, 10)
            g.savefig(path_to_zoom_metrics_plot)

    if len(evaluator.sample_functions) > 0 and evaluator.samples.shape[0] > 0:
        path_to_trace_plot = os.path.join(path_to_fig, "trace.png")
        logger.info("Plotting Sample Trace to {0}".format(path_to_trace_plot))
        plt.close('all')
        fig, axes = plot_trace_plot(evaluator)
        fig.set_size_inches(12, 10)
        fig.savefig(path_to_trace_plot)

        if len(evaluator.samples['iteration'].unique()) > 10:
            path_to_zoom_trace_plot = \
                os.path.join(path_to_fig, "trace_zoom.png")
            logger.info("Plotting Zoom Trace to {0}".format(
                path_to_zoom_trace_plot))
            plt.close('all')
            fig, axes = plot_trace_plot(evaluator, full_trace=False)
            fig.set_size_inches(12, 10)
            fig.savefig(path_to_zoom_trace_plot)

    return


def convert_gradient(gradient, grad_name):
    """Convert gradient with respect to one parameter
    to gradient with respect to A, Q, R.
    """
    # Try to convert gradient to scalar if it's not already
    if hasattr(gradient, 'shape') and gradient.shape != ():
        try:
            # Try to extract a scalar value from the gradient
            if hasattr(gradient, 'item'):
                gradient = gradient.item()  # Convert single-element array to scalar
            elif hasattr(gradient, 'flatten') and gradient.size > 0:
                gradient = gradient.flatten()[0]  # Take first element
            elif isinstance(gradient, list) and len(gradient) > 0:
                gradient = gradient[0]  # Take first element of list
        except (ValueError, IndexError, TypeError) as e:
            logger.warning(
                "Couldn't convert gradient to scalar: {}. Error: {}".format(gradient, e))
            gradient = 0.0  # Default to zero if conversion fails

    if grad_name == 'A':
        grad_phi = float(gradient) if not isinstance(
            gradient, (int, float)) else gradient
        grad_sigma = 0.0
        grad_tau = 0.0
    elif grad_name == 'Q':
        grad_phi = 0.0
        grad_sigma = float(gradient) if not isinstance(
            gradient, (int, float)) else gradient
        grad_tau = 0.0
    elif grad_name == 'R':
        grad_phi = 0.0
        grad_sigma = 0.0
        grad_tau = float(gradient) if not isinstance(
            gradient, (int, float)) else gradient
    else:
        raise Exception('Unknown grad name', grad_name)

    return np.array([grad_phi, grad_sigma, grad_tau])


###############################################################################
# Run Script ---------------------------------------------------------------
###############################################################################
if __name__ == '__main__':
    parser = construct_parser()
    logging.info("Parsing Args")
    args, extra = parser.parse_known_args()
    logging.info("Args: %s", str(args))
    if extra:
        logging.warning("Unused Arguments: {0}".format(extra))
    out = main(**vars(args))
    logging.info("..Complete")

# EOF
