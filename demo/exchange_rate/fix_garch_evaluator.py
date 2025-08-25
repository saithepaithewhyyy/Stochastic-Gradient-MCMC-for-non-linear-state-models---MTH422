"""
Helper script to create pickle-compatible evaluator files
"""
import pickle

# Try to use dill for better serialization
try:
    import dill
    use_dill = True
    print("Using dill for enhanced pickling support")
except ImportError:
    use_dill = False
    print("Warning: dill not found, using standard pickle.")
    print("To install: pip install dill")

# Create compatible evaluator class


class SimpleEvaluator:
    """A simple evaluator class that only contains parameters and sampler.
    This avoids pickling issues with custom metric functions.
    """

    def __init__(self, parameters_list, sampler):
        self.parameters_list = parameters_list
        self.sampler = sampler


def fix_garch_evaluator(result_dir):
    """Fix GARCH evaluator files for KSD calculation."""
    # Load original data and sampler
    try:
        with open(f"{result_dir}/data_and_sampler.pkl", "rb") as f:
            data = pickle.load(f)

        sampler = data['sampler']

        # Load parameter samples
        with open(f"{result_dir}/samples/sgld_samples_id0.pkl", "rb") as f:
            sgld_df = pickle.load(f)

        with open(f"{result_dir}/samples/ld_samples_id1.pkl", "rb") as f:
            ld_df = pickle.load(f)

        # Create new evaluator objects
        sgld_evaluator = SimpleEvaluator(
            parameters_list=sgld_df['parameters'].tolist(),
            sampler=sampler
        )

        ld_evaluator = SimpleEvaluator(
            parameters_list=ld_df['parameters'].tolist(),
            sampler=sampler
        )

        # Save with appropriate serializer
        if use_dill:
            with open(f"{result_dir}/sgld_evaluator.pkl", "wb") as f:
                dill.dump(sgld_evaluator, f)
            with open(f"{result_dir}/ld_evaluator.pkl", "wb") as f:
                dill.dump(ld_evaluator, f)
        else:
            with open(f"{result_dir}/sgld_evaluator.pkl", "wb") as f:
                pickle.dump(sgld_evaluator, f)
            with open(f"{result_dir}/ld_evaluator.pkl", "wb") as f:
                pickle.dump(ld_evaluator, f)

        print(f"Created compatible evaluator files in {result_dir}")
        print("Now you can run the KSD calculation script")

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def fix_svm_evaluator(result_dir):
    """Fix SVM evaluator files for KSD calculation."""
    # Same approach for SVM
    try:
        with open(f"{result_dir}/data_and_sampler.pkl", "rb") as f:
            data = pickle.load(f)

        sampler = data['sampler']

        with open(f"{result_dir}/samples/sgld_samples_id0.pkl", "rb") as f:
            sgld_df = pickle.load(f)

        with open(f"{result_dir}/samples/ld_samples_id1.pkl", "rb") as f:
            ld_df = pickle.load(f)

        sgld_evaluator = SimpleEvaluator(
            parameters_list=sgld_df['parameters'].tolist(),
            sampler=sampler
        )

        ld_evaluator = SimpleEvaluator(
            parameters_list=ld_df['parameters'].tolist(),
            sampler=sampler
        )

        if use_dill:
            with open(f"{result_dir}/sgld_evaluator.pkl", "wb") as f:
                dill.dump(sgld_evaluator, f)
            with open(f"{result_dir}/ld_evaluator.pkl", "wb") as f:
                dill.dump(ld_evaluator, f)
        else:
            with open(f"{result_dir}/sgld_evaluator.pkl", "wb") as f:
                pickle.dump(sgld_evaluator, f)
            with open(f"{result_dir}/ld_evaluator.pkl", "wb") as f:
                pickle.dump(ld_evaluator, f)

        print(f"Created compatible evaluator files in {result_dir}")

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Fix evaluator files for KSD calculation')
    parser.add_argument('--model_type', type=str, choices=['svm', 'garch'],
                        required=True, help='Model type (svm or garch)')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory containing model results')

    args = parser.parse_args()

    if args.model_type == 'garch':
        success = fix_garch_evaluator(args.result_dir)
    elif args.model_type == 'svm':
        success = fix_svm_evaluator(args.result_dir)

    if success:
        print("\nNow you can run KSD calculation:")
        print(f"python demo/exchange_rate/calculate_ksd.py "
              f"--model_type {args.model_type} --result_dir {args.result_dir}")
