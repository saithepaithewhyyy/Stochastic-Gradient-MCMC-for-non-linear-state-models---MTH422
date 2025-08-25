import numpy as np
import pandas as pd
import pickle
import os

from sgmcmc_ssm.models.garch import SeqGARCHSampler
from tqdm import tqdm

# Create output directory
output_folder = "eurgbp_garch_results"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(f"{output_folder}/samples", exist_ok=True)

# Load data
try:
    exchange_data = np.load('data/EURGBP_processed.npz')
    print("Loaded processed EURGBP data")
except (FileNotFoundError, IOError):
    print("Processing EURGBP data first")
    import subprocess
    subprocess.run(
        ["python", "demo/exchange_rate/process_gbp_data.py"], check=True)
    exchange_data = np.load('data/EURGBP_processed.npz')

print(list(exchange_data.keys()))
hourly_log_returns = exchange_data['hourly_log_returns']
hourly_dates = exchange_data['hourly_date']
print(f"Data shape: {hourly_log_returns.shape}")

# Scale data
observations = hourly_log_returns.reshape(-1, 1) * 1000

# Split data
gap_indices = np.where(np.diff(hourly_dates) > pd.Timedelta('6h'))[0].tolist()
split_observations = []
for start, end in zip([0]+gap_indices, gap_indices+[observations.size]):
    if end - start > 6:
        split_observations.append(observations[start:end])

# Save data info
with open(f"{output_folder}/data_info.pkl", "wb") as f:
    pickle.dump({
        "observations": split_observations,
        "hourly_dates": hourly_dates,
    }, f)

# Initialize GARCH model
sampler = SeqGARCHSampler(n=1, m=1, observations=split_observations)
sampler.prior_init()
sampler.project_parameters()

print(sampler.noisy_logjoint(kind='pf', pf='paris', N=10000,
                             return_loglike=True, tqdm=tqdm))

# For reproducing the results in the paper, use 8 hours (480 minutes)
fit_time = 60*60*8  # 8 hours

# Run SGLD with buffer=4 (buffer)
print("\nRunning SGLD with buffer=4")
sgld_parameters, sgld_time = sampler.fit_timed(
    iter_type='SGLD',
    epsilon=0.001, subsequence_length=16, num_sequences=1, buffer_length=4,
    kind='pf', pf_kwargs=dict(pf='poyiadjis_N', N=10000),
    max_time=fit_time,
    tqdm=tqdm,
)

# Save SGLD parameters
sgld_df = pd.DataFrame({"parameters": sgld_parameters, "time": sgld_time})
# Fix for parameters without get_free_state()
zero_grads = []
for _ in range(len(sgld_parameters)):
    zero_grads.append(np.zeros(4))  # GARCH has 4 parameters
sgld_df['grad'] = zero_grads
sgld_df['accept'] = 1.0
sgld_df['num_ksd_eval'] = 0.0

# Save to disk
sgld_df.to_pickle(f"{output_folder}/samples/sgld_samples_id0.pkl")
print(f"Saved SGLD parameters to {output_folder}/samples/sgld_samples_id0.pkl")

# Run LD (no buffer)
print("\nRunning LD (no buffer)")
sampler.parameters = sgld_parameters[0].copy()
ld_parameters, ld_time = sampler.fit_timed(
    iter_type='SGLD',
    epsilon=0.1, subsequence_length=-1, num_sequences=-1, buffer_length=0,
    kind='pf', pf_kwargs=dict(pf='paris', N=10000),
    max_time=fit_time,
    tqdm=tqdm,
)

# Save LD parameters
ld_df = pd.DataFrame({"parameters": ld_parameters, "time": ld_time})
# Fix for parameters without get_free_state()
zero_grads = []
for _ in range(len(ld_parameters)):
    zero_grads.append(np.zeros(4))  # GARCH has 4 parameters
ld_df['grad'] = zero_grads
ld_df['accept'] = 1.0
ld_df['num_ksd_eval'] = 0.0

# Save to disk
ld_df.to_pickle(f"{output_folder}/samples/ld_samples_id1.pkl")
print(f"Saved LD parameters to {output_folder}/samples/ld_samples_id1.pkl")

# Save original observations and sampler for KSD evaluation
with open(f"{output_folder}/data_and_sampler.pkl", "wb") as f:
    pickle.dump({
        "observations": split_observations,
        "sampler": sampler,
    }, f)

# Create evaluator objects that calculate_ksd.py expects


class Evaluator:
    def __init__(self, parameters_list, sampler):
        self.parameters_list = parameters_list
        self.sampler = sampler


# Save SGLD evaluator
sgld_evaluator = Evaluator(sgld_parameters, sampler)
with open(f"{output_folder}/sgld_evaluator.pkl", "wb") as f:
    pickle.dump(sgld_evaluator, f)

# Save LD evaluator
ld_evaluator = Evaluator(ld_parameters, sampler)
with open(f"{output_folder}/ld_evaluator.pkl", "wb") as f:
    pickle.dump(ld_evaluator, f)

print("\nAll files saved to", output_folder)
print("\nTo calculate KSD metrics, run:")
print("python demo/exchange_rate/calculate_ksd.py "
      f"--model_type garch --result_dir {output_folder}")
