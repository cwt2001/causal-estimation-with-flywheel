# Causal Estimation with Flywheel Effects

This repository provides code for the numerical experiments in the paper "Causal Estimation of Share-Induced Engagement with Flywheel Effects". The code includes data-generating processes, proposed and baseline estimators, single-setting and parameter-sweep runners, SLURM submission scripts, and notebooks for generating figures.

## Repository Structure

```text
configs/
  config.py                         # Runtime config for single-setting runs
  exp_config.py                     # Base config for parameter sweeps
  experiment_settings/              # Sweep definitions used by experiments.py
main.py                             # Run one configured setting
experiments.py                      # Run a parameter sweep setting
run_obs_window_sweep_experiments.py # Sweep the observation window W
run_rho0_scaling_experiments.py     # Sweep target rho0 values
estimators.py                       # Proposed and baseline estimators
utils.py                            # Network generation, sampling, simulation helpers
bayesian_bgps.py                    # Bayesian BGPS implementation
submit_slurm_experiments.sh         # Batch Slurm submitter for sweep settings
slurm_run_experiments.sbatch        # Slurm wrapper for experiments.py
slurm_sweep_obs_window.sbatch       # Slurm wrapper for W-window sweeps
slurm_sweep_rho0.sbatch             # Slurm wrapper for rho0 sweeps
display_results.ipynb               # Plotting and figure generation
requirements.txt                    # Python dependencies
results/                            # Experiment outputs
figures/                            # Generated figures
```

## Estimators: Paper to Code

| Method | Main output column | Main function |
|---|---|---|
| Proposed (Ours) | `proposed_*_GTE` | `proposed_estimator()` |
| DM | `naive_HT_*_GTE` | `naive_estimator_mean()` |
| DM-FO | `propagation_*_GTE` | `naive_estimator_mean()` |
| GCR | `cluster_mean_*_GTE` | `naive_estimator_mean_with_design()` |
| EW | `split_credit2_react_GTE` | `naive_estimator_split_credit2()` |
| HEW | `split_credit_react_GTE` | `naive_estimator_split_credit()` |
<!-- | BGPS | `bayesian_bgps_share_GTE` | `bayesian_bgps_estimator()` | -->

The code also writes ground-truth columns such as `truth_share_GTE`, which are used by `display_results.ipynb` when computing bias and MSE.

## Setup

Python 3.8+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

For cross-platform runs, set `MP_START_METHOD = None` in the config files unless you have a specific reason to override the platform default.

## Run One Setting for Debugging

Use this path when you want to run one configuration, usually as a quick test.

1. Edit `configs/config.py`.
   - `SIMULATION_METHOD` selects the simulator.
   - `NETWORK_PARAMS` selects the graph model.
   - `POPULATION_PARAMS` sets `n`, `m`, `pi`, and `p`.
   - `N_REPEATS` and `MAX_WORKERS` control runtime.

2. Run from the repository root.

Windows PowerShell:

```powershell
python main.py --config configs.config
```

Linux:

```bash
python main.py --config configs.config
```

The output is saved to:

```text
results/run_<timestamp>/
```

The folder contains files such as `settings.json`, `estimators.csv`, `estimator_summary.csv`, `poisson_results.csv`, and `inference_results.csv`.

## Main Paper Setting

This is the main public reproduction path for the numerical experiments in paper Sections 7.2 and 7.3. The setting is defined in `configs/experiment_settings/barabasi_uniform_sweep_disturbance.py`.

It matches the paper's synthetic simulation setup:

```text
graph: Barabasi-Albert
n: 50000
m: 2000
pi: 0.5
p: 0.5
disturbance levels Delta: 0.0, 0.3, 0.5, 0.7
```

The zero-disturbance case is included as a sanity check. The paper figures and coverage table use the nonzero disturbance levels `Delta = 0.3, 0.5, 0.7`.

### Run Locally

Run from the repository root:

```bash
python experiments.py --setting configs.experiment_settings.barabasi_uniform_sweep_disturbance --config configs.exp_config
```

This full setting is computationally heavy. For a smoke test, first lower `N_REPEATS` and `MAX_WORKERS` in `configs/exp_config.py`, lower `n` and `m` in `configs/experiment_settings/barabasi_uniform_sweep_disturbance.py`, or use `main.py` with `configs/config.py`.

Direct local runs write outputs to:

```text
results/experiments_<timestamp>/
```

The root contains `experiment_config.json`, `base_config.json`, and `all_results_summary.csv`. Each disturbance configuration has its own subfolder with per-repeat estimator and inference files.

### Run with Slurm

On a Slurm cluster, `submit_slurm_experiments.sh` is configured to submit the main paper setting by default.

Preview the command:

```bash
./submit_slurm_experiments.sh --dry-run
```

Submit the job:

```bash
./submit_slurm_experiments.sh
```

<!-- Useful options:

```bash
./submit_slurm_experiments.sh --partition amd
./submit_slurm_experiments.sh --partition intel,amd
./submit_slurm_experiments.sh --wait
./submit_slurm_experiments.sh --config configs.exp_config
``` -->

The submitter creates a shared root folder:

```text
results/experiments_<timestamp>/
```

The main paper setting writes to:

```text
results/experiments_<timestamp>/barabasi_uniform_sweep_disturbance/
```

You can also submit the setting directly:

```bash
sbatch slurm_run_experiments.sbatch --setting configs.experiment_settings.barabasi_uniform_sweep_disturbance --config configs.exp_config
```

### Plot Main Paper Results

After the main setting finishes, open `display_results.ipynb` and set:

```python
MODE = 'standard'
STANDARD_RESULTS_ROOT = RESULTS_DIR / 'experiments_<timestamp>'
SELECTED_SETTINGS = [
    'barabasi_uniform_sweep_disturbance',
]
```

Then run the notebook cells. Generated figures are saved to:

```text
figures/<experiment_root>/barabasi_uniform_sweep_disturbance/
```

### Match Outputs to the Paper

Use `all_results_summary.csv` and `display_results.ipynb` to connect simulation outputs to the paper:

| Paper result | Repository output |
|---|---|
| Figure 1, impact-of-sharing estimates | `figures/<experiment_root>/barabasi_uniform_sweep_disturbance/*_GTE_estimate_boxplot.png` |
| Figure 1, impact-of-sharing MSE | `figures/<experiment_root>/barabasi_uniform_sweep_disturbance/*_GTE_mse.png` |
| Figure 2, reactivation-rate estimates | `figures/<experiment_root>/barabasi_uniform_sweep_disturbance/*_GTE_ra_estimate_boxplot.png` |
| Figure 2, reactivation-rate MSE | `figures/<experiment_root>/barabasi_uniform_sweep_disturbance/*_GTE_ra_mse.png` |
| Table 1, 95% coverage | `coverage_rate` in `all_results_summary.csv` for `Delta = 0.3, 0.5, 0.7` |

The paper's Figure 3 and Table 2 are not reproduced by this public repository.

## Extension Experiments

Extension experiments include all non-main settings: robustness sweeps, alternate parameter distributions, BGPS, observation-window sweeps, and `rho0` scaling.

### Standard Robustness Sweeps

Use `experiments.py` when you want to run a sweep over graph settings, disturbance levels, exposure probabilities, or parameter distributions.

1. Choose or edit a setting module under `configs/experiment_settings/`.
2. Keep shared simulation parameters in `configs/exp_config.py`.
3. Run:

```bash
python experiments.py --setting configs.experiment_settings.barabasi_uniform_sweep_pi --config configs.exp_config
```

Replace `barabasi_uniform_sweep_pi` with any extension setting module name under `configs/experiment_settings/`, for example:

```text
barabasi_gamma_sweep_disturbance
barabasi_lognormal_sweep_disturbance
barabasi_uniform_more_shares
barabasi_uniform_sweep_degree
barabasi_uniform_sweep_n
barabasi_uniform_sweep_pi
sbm_uniform_sweep_block_num
watts_uniform_sweep_degree
watts_uniform_sweep_rewiring_p
```

Direct sweep runs write outputs to:

```text
results/experiments_<timestamp>/
```

Each configuration has its own subfolder. The sweep root also contains `experiment_config.json`, `base_config.json`, and `all_results_summary.csv`.

### Submit Extension Sweeps with Slurm

On a Slurm cluster, use `submit_slurm_experiments.sh` to submit several sweep settings.

1. Edit the `JOBS` array in `submit_slurm_experiments.sh`.
   - Each line has the format `setting_module|nodelist`.
   - Leave `nodelist` empty to let Slurm choose a node.

2. Preview commands without submitting:

```bash
./submit_slurm_experiments.sh --dry-run
```

3. Submit jobs:

```bash
./submit_slurm_experiments.sh
```

<!-- Useful options:

```bash
./submit_slurm_experiments.sh --partition amd
./submit_slurm_experiments.sh --partition intel,amd
./submit_slurm_experiments.sh --wait
./submit_slurm_experiments.sh --config configs.exp_config
``` -->

`submit_slurm_experiments.sh` creates one shared root folder:

```text
results/experiments_<timestamp>/
```

Each submitted setting writes to:

```text
results/experiments_<timestamp>/<setting_name>/
```

You can also submit one setting directly:

```bash
sbatch slurm_run_experiments.sbatch --setting configs.experiment_settings.barabasi_uniform_sweep_pi --config configs.exp_config
```

### BGPS

BGPS is adapted from *Estimating Causal Effects under Network Interference with Bayesian Generalized Propensity Scores* (Forastiere et al., JMLR 2022) and tailored to Bernoulli randomized experiments. BGPS is disabled by default; enable it with `USE_BAYESIAN_BGPS=1` and run the BGPS setting.

Linux:

```bash
USE_BAYESIAN_BGPS=1 python experiments.py --setting configs.experiment_settings.bgps_setting --config configs.exp_config
```

Windows PowerShell:

```powershell
$env:USE_BAYESIAN_BGPS = "1"
python experiments.py --setting configs.experiment_settings.bgps_setting --config configs.exp_config
Remove-Item Env:\USE_BAYESIAN_BGPS
```

With Slurm:

```bash
sbatch --export=ALL,USE_BAYESIAN_BGPS=1 slurm_run_experiments.sbatch --setting configs.experiment_settings.bgps_setting --config configs.exp_config
```

The Bayesian dependencies are listed in `requirements.txt`. If you do not need BGPS, leave `USE_BAYESIAN_BGPS` unset.

### Scaling the Spectral Radius $\rho(Q^{(k)})$ 

Use this runner to rescale sharing parameters to target `rho0` values.

```bash
python run_rho0_scaling_experiments.py --config configs.config --n 5000 --m-edges 20 --max-workers 100 --targets 0.05,0.2,0.5,0.8
```

Useful options:

```bash
python run_rho0_scaling_experiments.py --config configs.config --n 5000 --m-edges 20 --max-workers 100 --targets 0.2,0.5 --no-baseline
python run_rho0_scaling_experiments.py --config configs.config --n 5000 --m-edges 20 --max-workers 100 --targets 0.05,0.2,0.5,0.8 --quiet
```

With Slurm:

```bash
sbatch slurm_sweep_rho0.sbatch
```

Outputs are saved to `results/experiments_<timestamp>/` with subfolders named by target and actual `rho0`.


### Varying Observation Window W

Use this runner to study the delayed observation window `W`.

```bash
python run_obs_window_sweep_experiments.py --config configs.config --n 5000 --m-edges 20 --max-workers 10 --w-values 0,5,10,24,100
```

With Slurm:

```bash
sbatch slurm_sweep_obs_window.sbatch
```

Outputs are saved to `results/experiments_<timestamp>/` with subfolders such as `W_0`, `W_5`, and `W_100`.



### Plot Extension Results

Use `display_results.ipynb` after experiments finish.

For standard robustness sweeps from `experiments.py`:

1. Open `display_results.ipynb`.
2. Set:

```python
MODE = 'standard'
STANDARD_RESULTS_ROOT = RESULTS_DIR / 'experiments_<timestamp>'
SELECTED_SETTINGS = [
    'barabasi_uniform_sweep_pi',
    'watts_uniform_sweep_degree',
]
```

3. Run the notebook cells.

Use setting names that match the subfolders under `STANDARD_RESULTS_ROOT`.

For manually curated folders, use manual mode:

```python
MODE = 'manual'
MANUAL_PRESET = 'rho0'      # choose 'rho0', 'W', or 'bayesian'
```

Then update the corresponding entry in `MANUAL_SPECS`:

```python
MANUAL_SPECS['rho0']['exp_dir'] = 'experiments_<timestamp>'
MANUAL_SPECS['rho0']['folders'] = [
    ('rho0_target_0p05__actual_0p05__scale_0p3754', 0.05, '0.05'),
    ('rho0_target_0p2__actual_0p2__scale_1p5016', 0.2, '0.2'),
]
```

For `W`, update `MANUAL_SPECS['W']['exp_dir']` and folders such as `W_0`, `W_5`, and `W_100`.

For BGPS, update `MANUAL_SPECS['bayesian']['exp_dir']` and use the subfolder names produced by `bgps_setting`.

Generated figures are saved to:

```text
figures/<experiment_root>/<setting>/
```

The notebook saves grouped estimate or bias boxplots and MSE curves.

## Output Files

Most experiment folders include:

```text
estimators.csv                 # Per-repeat estimator values
estimator_summary.csv          # Mean and standard error for each estimator
poisson_results.csv            # Per-repeat Poisson diagnostics
poisson_results_summary.json   # Poisson diagnostic summary
inference_results.csv          # Per-repeat inference results
inference_summary.json         # Coverage and interval summaries
```

Sweep roots also include:

```text
all_results_summary.csv
experiment_config.json
base_config.json
```

These files are the inputs expected by `display_results.ipynb`.

## Quick Checks

Show command-line options:

```bash
python main.py --help
python experiments.py --help
python run_obs_window_sweep_experiments.py --help
python run_rho0_scaling_experiments.py --help
```

Run a small single-setting test by lowering `N_REPEATS`,`MAX_WORKERS`, `n`, and `m` in `configs/config.py`, then running:

```bash
python main.py --config configs.config
```
