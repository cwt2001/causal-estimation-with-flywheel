import os
from datetime import datetime
import json
import time
import copy
import argparse
import traceback

import pandas as pd
import numpy as np

from utils import generate_network, sample_watching_params, sample_sharing_params
from utils import diagnose_poisson_columns
from utils import prepare_cluster_randomization
from configs.loader import load_config_module, require_config_attrs
from log_capture import start_run_log_capture

from main import run_simulations_parallel


def save_results_to_folder(all_results, make_inference, run_dir,
                           print_results_summary=True, print_poisson_results=True):

    df_estimators = all_results['df_estimators']
    watched_by_share_num_mat = all_results['watched_by_share_num_mat']
    df_inference = all_results['df_inference']

    n_repeats = df_estimators.shape[0]

    estimator_cols = list(df_estimators.columns)
    means = df_estimators[estimator_cols].mean()
    se = df_estimators[estimator_cols].std(ddof=1) / np.sqrt(n_repeats)
    df_summary = pd.DataFrame({
        'mean': means,
        'se': se
    })

    if print_results_summary:
        print("\nEstimator Summary Statistics")
        print("="*50)
        print(df_summary.to_string(float_format=lambda x: f"{x:.6f}"))
        print("="*50)

    if make_inference:
        inference_type = df_inference['type'].iloc[0]
        inference_summary = {
            "inference_type": inference_type,
            "n_repeats": n_repeats,
            "coverage_rate": float(df_inference['is_cover_truth'].mean())
        }

        if inference_type == 'A/A':
            inference_summary["mean_GTE_truth"] = 0.0
        else:
            inference_summary["mean_GTE_truth"] = float(np.mean(df_inference['GTE_truth']))
            inference_summary['mean_estimate'] = float(np.mean(df_inference['estimate']))

        print("\n=== Inference Summary ===")
        print(f"Inference Type: {inference_summary['inference_type']}")
        print(f"Number of Repeats: {inference_summary['n_repeats']}")
        if inference_type == 'A/B':
            print(f"Mean GTE Truth: {inference_summary['mean_GTE_truth']:.6f}")
            print(f"Mean Estimate: {inference_summary['mean_estimate']:.6f}")
        else:
            print(f"Mean GTE Truth: {inference_summary['mean_GTE_truth']:.6f}")
        print(f"Coverage Rate: {inference_summary['coverage_rate']*100:.2f}%\n")

    # Poisson goodness-of-fit test
    alpha = 0.05
    n = watched_by_share_num_mat.shape[1]
    poisson_results = diagnose_poisson_columns(watched_by_share_num_mat, alpha=alpha)
    good_mean_var_rate = np.sum(poisson_results['good_mean_var'])/n
    avg_diff_mean_var = np.mean(poisson_results['mean_var_diff'])
    avg_ratio_mean_var = np.mean(poisson_results['mean_var_ratio'])
    valid_tests_rate = np.sum(poisson_results['valid_tests'])/n
    poisson_like_rate = np.sum(poisson_results['poisson_like'])/np.sum(poisson_results['valid_tests']) if np.sum(poisson_results['valid_tests']) > 0 else np.nan

    poisson_results_summary = {
        'good_mean_var_rate': float(good_mean_var_rate),
        'avg_diff_mean_var': float(avg_diff_mean_var),
        'avg_ratio_mean_var': float(avg_ratio_mean_var),
        'valid_tests_rate': float(valid_tests_rate),
        'poisson_like_rate': float(poisson_like_rate) if not np.isnan(poisson_like_rate) else None
    }

    if print_poisson_results:
        print(f"=== Poisson Distribution Diagnostic (n_repeats={n_repeats}, n={n}) ===")
        print(f"--- Summary Statistics ---")
        print(f"Mean-Variance Agreement:")
        print(f"  Columns with |var/mean - 1| < 0.1: {np.sum(poisson_results['good_mean_var'])}/{n} ({100*good_mean_var_rate:.1f}%)")
        print(f"  Avg |var - mean|: {avg_diff_mean_var:.4f}")
        print(f"  Avg var/mean ratio: {avg_ratio_mean_var:.4f}")

        if np.sum(poisson_results['valid_tests']) > 0:
            print(f"\nGoodness-of-Fit Tests (alpha={alpha}):")
            print(f"  Valid tests: {np.sum(poisson_results['valid_tests'])}/{n} ({100*valid_tests_rate:.1f}%)")
            print(f"  Columns passing Poisson test: {np.sum(poisson_results['poisson_like'])}/{np.sum(poisson_results['valid_tests'])} ({100*poisson_like_rate:.1f}%)")

    # Save results
    os.makedirs(run_dir, exist_ok=True)

    # Save estimator results
    estimators_csv_path = os.path.join(run_dir, 'estimators.csv')
    df_estimators.to_csv(estimators_csv_path, index_label='repeat_id')

    summary_csv_path = os.path.join(run_dir, 'estimator_summary.csv')
    df_summary.to_csv(summary_csv_path, index_label='estimator')

    # Save Poisson results
    poisson_results_csv_path = os.path.join(run_dir, 'poisson_results.csv')
    poisson_results.to_csv(poisson_results_csv_path, index_label='user_id')

    poisson_results_summary_path = os.path.join(run_dir, "poisson_results_summary.json")
    with open(poisson_results_summary_path, "w", encoding="utf-8") as f:
        json.dump(poisson_results_summary, f, ensure_ascii=False, indent=2)

    # Save inference results
    if make_inference:
        inference_csv_path = os.path.join(run_dir, 'inference_results.csv')
        df_inference.to_csv(inference_csv_path, index_label='repeat_id')

        inference_summary_path = os.path.join(run_dir, 'inference_summary.json')
        with open(inference_summary_path, "w", encoding="utf-8") as f:
            json.dump(inference_summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {run_dir}")

    return df_summary, poisson_results_summary, inference_summary if make_inference else None


def _format_param_token(value):
    """Format values into compact, stable tokens for folder names."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


def _build_c_by_param_suffix(c_by_param):
    """Build a deterministic suffix from c_by_param to avoid folder collisions."""
    if not isinstance(c_by_param, dict) or len(c_by_param) == 0:
        return None

    key_alias = {
        'a_d': 'ad',
        'b_d': 'bd',
        'phi': 'phi',
        'varphi': 'var',
        'theta': 'th',
    }
    preferred_order = ['a_d', 'b_d', 'phi', 'varphi', 'theta']

    parts = []
    for key in preferred_order:
        if key in c_by_param:
            parts.append(f"{key_alias[key]}{_format_param_token(c_by_param[key])}")

    for key in sorted(k for k in c_by_param.keys() if k not in preferred_order):
        parts.append(f"{key}{_format_param_token(c_by_param[key])}")

    if not parts:
        return None

    return "c_" + "_".join(parts)


def run_all_experiments(experiment_setting=None,
                        base_config_path=None):
    """
    Run experiments for all parameter combinations.

    Parameters
    ----------
    experiment_setting : dict, optional
        One experiment setting dictionary that contains base_config_path and sweep variables.
    base_config_path : str
        Base config module path. If provided, overrides experiment_setting['base_config_path'].
    """
    setting_dict = copy.deepcopy(experiment_setting) if experiment_setting else {}

    if base_config_path is None:
        base_config_path = setting_dict.get('base_config_path',None)
    if base_config_path is None:
        base_config_path = 'configs.exp_config'

    config_module, config_source = load_config_module(base_config_path)
    config_values = require_config_attrs(
        config_module,
        [
            'SIMULATION_METHOD', 'T_MAX', 'N_REPEATS', 'MAX_WORKERS',
            'POPULATION_PARAMS', 'PARAMS_RELATED', 'RANDOM_SEEDS', 'HOMO_VALUES',
            'HETERO_RANGES', 'ESTIMATOR_TYPE', 'MP_START_METHOD', 'CLUSTER_RANDOMIZATION'
        ],
        source_label=base_config_path
    )

    SIMULATION_METHOD = config_values['SIMULATION_METHOD']
    T_MAX = config_values['T_MAX']
    N_REPEATS = config_values['N_REPEATS']
    MAX_WORKERS = config_values['MAX_WORKERS']
    POPULATION_PARAMS = config_values['POPULATION_PARAMS']
    PARAMS_RELATED = config_values['PARAMS_RELATED']
    RANDOM_SEEDS = config_values['RANDOM_SEEDS']
    HOMO_VALUES = config_values['HOMO_VALUES']
    HETERO_RANGES = config_values['HETERO_RANGES']
    ESTIMATOR_TYPE = config_values['ESTIMATOR_TYPE']
    MP_START_METHOD = config_values['MP_START_METHOD']
    CLUSTER_RANDOMIZATION = config_values['CLUSTER_RANDOMIZATION']
    HAWKES_CT_WINDOW = getattr(config_module, 'HAWKES_CT_WINDOW', {})

    if not isinstance(HAWKES_CT_WINDOW, dict):
        raise TypeError("HAWKES_CT_WINDOW must be a dict when provided")

    hawkes_ct_window_W = float(HAWKES_CT_WINDOW.get('W', 0.0))
    hawkes_ct_beta = float(HAWKES_CT_WINDOW.get('beta', 1.0))
    hawkes_ct_simulate_beyond_window = HAWKES_CT_WINDOW.get('simulate_beyond_window', False)
    if not isinstance(hawkes_ct_simulate_beyond_window, bool):
        raise TypeError("HAWKES_CT_WINDOW['simulate_beyond_window'] must be a bool when provided")

    # Backup base config
    base_config = {
        'SIMULATION_METHOD': SIMULATION_METHOD,
        'T_MAX': T_MAX,
        'N_REPEATS': N_REPEATS,
        'MAX_WORKERS': MAX_WORKERS,
        'POPULATION_PARAMS': POPULATION_PARAMS,
        'PARAMS_RELATED': PARAMS_RELATED,
        'RANDOM_SEEDS': RANDOM_SEEDS,
        'HOMO_VALUES': HOMO_VALUES,
        'HETERO_RANGES': HETERO_RANGES,
        'ESTIMATOR_TYPE': ESTIMATOR_TYPE,
        'MP_START_METHOD': MP_START_METHOD,
        'CLUSTER_RANDOMIZATION': CLUSTER_RANDOMIZATION,
        'HAWKES_CT_WINDOW': {
            'W': hawkes_ct_window_W,
            'beta': hawkes_ct_beta,
            'simulate_beyond_window': hawkes_ct_simulate_beyond_window,
            'T_eval': T_MAX + hawkes_ct_window_W,
        }
    }
    
    # Base parameters
    p = POPULATION_PARAMS['p']
    population_defaults = POPULATION_PARAMS if isinstance(POPULATION_PARAMS, dict) else {}
    default_n = population_defaults.get('n', 100)
    default_m = population_defaults.get('m', 200)
    default_pi = population_defaults.get('pi', 0.5)
    T_max = T_MAX
    n_repeats = N_REPEATS
    max_workers = MAX_WORKERS
    seed_graph = RANDOM_SEEDS['seed_graph']
    seed_params = RANDOM_SEEDS['seed_params']
    seed_base = RANDOM_SEEDS['seed_base']
    make_inference = True
    
    # Resolve parameter combinations from experiment_setting, then defaults.
    m_configs = setting_dict.get('m_configs')
    if m_configs is None:
        m_configs = [{'m': default_m, 'name': f"m{default_m}"}]

    graph_configs = setting_dict.get('graph_configs')
    if graph_configs is None:
        graph_configs = [
            {
                'n': default_n,
                'method': 'regular',
                'm_edges': 10,
                'graph_p': None,
                'name': f"n{default_n}_regular_d10"
            }
        ]

    phi_varphi_theta_shifts = setting_dict.get('phi_varphi_theta_shifts')
    if phi_varphi_theta_shifts is None:
        phi_varphi_theta_shifts = [
            {'phi': (0, 0), 'varphi': (0, 0), 'theta': (0, 0), 'name': 'phi_0_0_var_0_0_d_0_0'}
        ]

    pi_values = setting_dict.get('pi_values')
    if pi_values is None:
        pi_values = [default_pi]

    random_para_dist_configs = setting_dict.get('random_para_dist_configs')
    if random_para_dist_configs is None:
        random_para_dist_configs = [
            {'name': 'legacy_uniform_ranges', 'dist': None, 'c_by_param': None}
        ]
    if not isinstance(random_para_dist_configs, list):
        raise TypeError("random_para_dist_configs must be a list of dicts or None")

    setting_name = setting_dict.get('name', 'manual_setting')
    
    # Create main experiment folder.
    # When launched by submit_slurm_experiments.sh, FW_RESULTS_ROOT/FW_SETTING_SUBDIR
    # point to the shared run root and per-setting subdirectory.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fw_results_root = os.getenv("FW_RESULTS_ROOT", "").strip()
    fw_setting_subdir = os.getenv("FW_SETTING_SUBDIR", "").strip()

    if fw_results_root:
        if fw_setting_subdir:
            main_exp_dir = os.path.join(fw_results_root, fw_setting_subdir)
        else:
            main_exp_dir = fw_results_root
        # Avoid clashing with Slurm files named experiments_<jobid>.out/.err.
        run_log_prefix = "runlog"
    else:
        main_exp_dir = os.path.join("results", f"experiments_{ts}")
        run_log_prefix = "experiments"

    os.makedirs(main_exp_dir, exist_ok=True)
    run_log_capture = start_run_log_capture(main_exp_dir, prefix=run_log_prefix)
    
    # Save experiment config
    exp_meta = {
        "timestamp": ts,
        "run_logs": {
            "stdout": os.path.basename(run_log_capture.stdout_path),
            "stderr": os.path.basename(run_log_capture.stderr_path),
        },
        "base_parameters": {
            "simulation_method": SIMULATION_METHOD,
            "T_max": T_max,
            "n_repeats": n_repeats,
            "max_workers": max_workers,
            "estimator_type": ESTIMATOR_TYPE,
            "mp_start_method": MP_START_METHOD,
            "random_seeds": RANDOM_SEEDS,
            "cluster_randomization": CLUSTER_RANDOMIZATION,
            "hawkes_ct_window": {
                "W": hawkes_ct_window_W,
                "beta": hawkes_ct_beta,
                "simulate_beyond_window": hawkes_ct_simulate_beyond_window,
                "T_eval": T_max + hawkes_ct_window_W,
            },
            "population_defaults": {
                "n": default_n,
                "m": default_m,
                "pi": default_pi,
                "p": p,
            },
            "make_inference": make_inference    
        },
        "experiment_setting": {
            "experiment_setting_name": setting_name,
            "m_configs": m_configs,
            "graph_configs": graph_configs,
            "phi_varphi_theta_shifts": phi_varphi_theta_shifts,
            "pi_values": pi_values,
            "random_para_dist_configs": random_para_dist_configs,
        }
    }
    
    with open(os.path.join(main_exp_dir, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(exp_meta, f, ensure_ascii=False, indent=2)
    
    # Save imported base config backup
    with open(os.path.join(main_exp_dir, "base_config.json"), "w", encoding="utf-8") as f:
        json.dump(base_config, f, ensure_ascii=False, indent=2)
    
    # Total configurations
    total_configs = (
        len(graph_configs)
        * len(m_configs)
        * len(phi_varphi_theta_shifts)
        * len(pi_values)
        * len(random_para_dist_configs)
    )
    print(f"Total number of configurations: {total_configs}")
    print(f"Results will be saved to: {main_exp_dir}\n")
    
    config_counter = 0
    failed_configs = 0
    all_results_summary = []
    
    # Iterate over all parameter combinations
    for dist_cfg in random_para_dist_configs:
        if not isinstance(dist_cfg, dict):
            raise TypeError("Each random_para_dist config must be a dict")

        dist_name = dist_cfg.get('dist')
        if isinstance(dist_name, str):
            dist_name = dist_name.strip().lower()
        dist_label = dist_cfg.get('name', 'legacy_uniform_ranges')
        if dist_label is None:
            dist_label = 'legacy_uniform_ranges'
        if not isinstance(dist_label, str):
            dist_label = str(dist_label)
        dist_label = dist_label.strip() or 'legacy_uniform_ranges'
        c_by_param = dist_cfg.get('c_by_param')
        c_by_param_suffix = _build_c_by_param_suffix(c_by_param)

        for m_cfg in m_configs:
            for pvt_cfg in phi_varphi_theta_shifts:
                for graph_cfg in graph_configs:
                    for pi in pi_values:
                        config_counter += 1
                        n = graph_cfg['n']
                        m = m_cfg['m']

                        print(f"\n{'='*80}")
                        print(f"Configuration {config_counter}/{total_configs}")
                        print(
                            f"m={m}, phi_shift={pvt_cfg['name']}, Graph: {graph_cfg['name']}, "
                            f"Pi: {pi}, Dist: {dist_label}"
                        )
                        print(f"{'='*80}")

                        safe_dist_label = dist_label.replace(' ', '_').replace('/', '-')
                        safe_c_by_param_suffix = None
                        if c_by_param_suffix:
                            safe_c_by_param_suffix = (
                                c_by_param_suffix
                                .replace(' ', '_')
                                .replace('/', '-')
                                .replace(':', '-')
                            )

                        # Create subfolder name
                        subfolder_parts = [
                            graph_cfg['name'],
                            m_cfg['name'],
                            pvt_cfg['name'],
                            f"pi{pi:.1f}",
                            safe_dist_label,
                        ]
                        if safe_c_by_param_suffix:
                            subfolder_parts.append(safe_c_by_param_suffix)

                        subfolder_name = "_".join(subfolder_parts).replace('.', 'p')
                        run_dir = os.path.join(main_exp_dir, subfolder_name)

                        try:
                            # Generate network
                            graph_method = (graph_cfg.get('method') or '').lower()
                            if graph_method in ['fbsnd', 'brightkite', 'twitch_gamer']:
                                G = generate_network(n, method=graph_cfg['method'])
                                actual_n = len(G.nodes())
                                print(f"Loaded real social network with {actual_n} nodes")
                            else:
                                network_kwargs = {
                                    key: value for key, value in graph_cfg.items()
                                    if key not in {'name', 'n'}
                                }
                                G = generate_network(
                                    n,
                                    seed=seed_graph,
                                    **network_kwargs
                                )
                                actual_n = n

                            # Cluster partition is computed once for this graph configuration.
                            cluster_labels, cluster_metadata = prepare_cluster_randomization(
                                G,
                                actual_n,
                                cluster_randomization_config=CLUSTER_RANDOMIZATION
                            )
                            print(
                                f"Cluster randomization: method={cluster_metadata['method']}, "
                                f"n_clusters={cluster_metadata['n_clusters']}, "
                                f"size[min/median/max]={cluster_metadata['cluster_size_min']}/"
                                f"{cluster_metadata['cluster_size_median']:.1f}/"
                                f"{cluster_metadata['cluster_size_max']}"
                            )

                            is_homo = PARAMS_RELATED['is_homo']
                            phi_shift_min, phi_shift_max = pvt_cfg['phi']
                            varphi_shift_min, varphi_shift_max = pvt_cfg['varphi']
                            theta_shift_min, theta_shift_max = pvt_cfg['theta']

                            if is_homo:
                                theta_shift = (theta_shift_max + theta_shift_min) / 2
                                phi_shift = (phi_shift_max + phi_shift_min) / 2
                                varphi_shift = (varphi_shift_max + varphi_shift_min) / 2

                                homo_values = copy.deepcopy(HOMO_VALUES)
                                homo_values['sharing_offsets']['phi'] = {'C': phi_shift, 'O': phi_shift}
                                homo_values['sharing_offsets']['varphi'] = {'C': varphi_shift, 'O': varphi_shift}
                                homo_values['sharing_offsets']['theta'] = {'C': theta_shift, 'O': theta_shift}

                                hetero_ranges = {
                                    'watching': None,
                                    'sharing_T': None,
                                    'sharing_ds_perturbation': None,
                                    'sharing_shift_ranges': None
                                }
                            else:
                                # Random perturbation can be added to group O
                                hetero_ranges = copy.deepcopy(HETERO_RANGES)
                                hetero_ranges['sharing_shift_ranges']['phi'] = {
                                    'C': (phi_shift_min, phi_shift_max),
                                    'O': (phi_shift_min, phi_shift_max)
                                }
                                hetero_ranges['sharing_shift_ranges']['varphi'] = {
                                    'C': (varphi_shift_min, varphi_shift_max),
                                    'O': (varphi_shift_min, varphi_shift_max)
                                }
                                hetero_ranges['sharing_shift_ranges']['theta'] = {
                                    'C': (theta_shift_min, theta_shift_max),
                                    'O': (theta_shift_min, theta_shift_max)
                                }

                                homo_values = {
                                    'watching': None,
                                    'sharing_T': None,
                                    'sharing_ds_perturbation': None,
                                    'sharing_offsets': None
                                }

                            watching_params = sample_watching_params(
                                actual_n, m,
                                seed=seed_params,
                                homo=is_homo,
                                homo_values=homo_values['watching'],
                                hetero_ranges=hetero_ranges['watching'],
                                random_para_dist=dist_name,
                                c_by_param=c_by_param
                            )

                            sharing_params = sample_sharing_params(
                                actual_n, m,
                                seed=seed_params,
                                homo=is_homo,
                                homo_values_T=homo_values['sharing_T'],
                                homo_ds_perturbation=homo_values['sharing_ds_perturbation'],
                                homo_offsets=homo_values['sharing_offsets'],
                                hetero_ranges_T=hetero_ranges['sharing_T'],
                                hetero_ds_perturbation=hetero_ranges['sharing_ds_perturbation'],
                                hetero_shift_ranges=hetero_ranges['sharing_shift_ranges'],
                                random_para_dist=dist_name,
                                c_by_param=c_by_param
                            )

                            start_time = time.time()
                            all_results = run_simulations_parallel(
                                n_repeats=n_repeats,
                                G=G, n=actual_n, m=m, T_max=T_max,
                                watching_params=watching_params,
                                sharing_params=sharing_params,
                                pi=pi, p=p,
                                simulation_method=SIMULATION_METHOD,
                                inference=make_inference,
                                estimator_type=ESTIMATOR_TYPE,
                                cluster_baseline_labels=cluster_labels,
                                hawkes_ct_window_W=hawkes_ct_window_W,
                                hawkes_ct_beta=hawkes_ct_beta,
                                hawkes_ct_simulate_beyond_window=hawkes_ct_simulate_beyond_window,
                                seed_base=seed_base,
                                max_workers=max_workers,
                                mp_start_method=MP_START_METHOD
                            )
                            elapsed_time = time.time() - start_time

                            print(f"Time cost: {elapsed_time:.2f} seconds")

                            df_summary, _, inference_summary = save_results_to_folder(
                                all_results, make_inference, run_dir,
                                print_results_summary=True,
                                print_poisson_results=True
                            )

                            # Record to summary
                            result_record = {
                                'config_id': config_counter,
                                'n': actual_n,
                                'm': m,
                                'graph': graph_cfg['name'],
                                'phi_varphi_theta_shift': pvt_cfg['name'],
                                'pi': pi,
                                'random_para_dist_name': dist_label,
                                'random_para_dist': dist_name if dist_name is not None else 'legacy_uniform_ranges',
                                'random_para_c_by_param': json.dumps(c_by_param) if c_by_param is not None else None,
                                'folder': subfolder_name,
                                'elapsed_time': elapsed_time,
                                'randomization_unit': cluster_metadata['unit'],
                                'cluster_method': cluster_metadata['method'],
                                'n_clusters': cluster_metadata['n_clusters']
                            }

                            # Add estimator means and standard errors
                            for estimator in df_summary.index:
                                result_record[f'{estimator}_mean'] = df_summary.loc[estimator, 'mean']
                                result_record[f'{estimator}_se'] = df_summary.loc[estimator, 'se']

                            # Add inference results
                            if make_inference and inference_summary:
                                result_record['coverage_rate'] = inference_summary['coverage_rate']

                            all_results_summary.append(result_record)

                        except Exception as e:
                            failed_configs += 1
                            error_message = f"{type(e).__name__}: {e}"
                            print(f"ERROR in configuration {config_counter}: {error_message}")
                            print(traceback.format_exc())
                            all_results_summary.append({
                                'config_id': config_counter,
                                'n': graph_cfg.get('n'),
                                'm': m_cfg.get('m'),
                                'graph': graph_cfg.get('name'),
                                'phi_varphi_theta_shift': pvt_cfg.get('name'),
                                'pi': pi,
                                'random_para_dist_name': dist_label,
                                'status': 'error',
                                'error': error_message
                            })

    # Save all results summary
    df_all_results = pd.DataFrame(all_results_summary)
    df_all_results.to_csv(os.path.join(main_exp_dir, 'all_results_summary.csv'), index=False)
    
    print(f"\n{'='*80}")
    if failed_configs == 0:
        print(f"All experiments completed!")
    else:
        print(f"Experiments completed with {failed_configs} failed configuration(s).")
    print(f"Results saved to: {main_exp_dir}")
    print(f"{'='*80}")
    
    run_log_capture.close()
    return main_exp_dir, df_all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run parameter sweep experiments with setting'
    )
    parser.add_argument(
        '--setting',
        default='configs.experiment_settings.bgps_setting',
        help=(
            'Experiment setting module path. '
            'The module must define EXPERIMENT_SETTING dict '
            '(e.g. configs.experiment_settings.default_setting).'
        )
    )
    parser.add_argument(
        '--config',
        default='configs.exp_config',
        help=(
            'Optional base config module path override '
            '(e.g. configs.exp_config). When provided, it overrides '
            'base_config_path in EXPERIMENT_SETTING.'
        )
    )
    args = parser.parse_args()

    experiment_setting = None
    if args.setting:
        setting_module, _ = load_config_module(args.setting)
        setting_values = require_config_attrs(
            setting_module,
            ['EXPERIMENT_SETTING'],
            source_label=args.setting
        )
        experiment_setting = copy.deepcopy(setting_values['EXPERIMENT_SETTING'])
        if not isinstance(experiment_setting, dict):
            raise TypeError("EXPERIMENT_SETTING must be a dict")

    main_exp_dir, df_summary = run_all_experiments(
        experiment_setting=experiment_setting,
        base_config_path=args.config
    )
