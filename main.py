import time
from datetime import datetime
import os
import json
import argparse
from scipy import stats

import pandas as pd
import numpy as np

import platform
from tqdm import tqdm
import multiprocessing as mp

from utils import sample_watching_params, sample_sharing_params
from utils import generate_network
from utils import (
    simulate_social_sharing_discrete,
    simulate_social_sharing_hawkes,
    simulate_social_sharing_hawkes_ct_exp_window,
)
from utils import diagnose_poisson_columns
from utils import check_if_aa
from utils import prepare_cluster_randomization
from utils import compute_rho0_from_sharing_params

from estimators import proposed_estimator, ground_truth
from estimators import proposed2_estimator, naive_estimator_split_credit, naive_estimator_split_credit2, naive_estimator_filter_users
from estimators import naive_estimator_mean_with_design
from estimators import bayesian_bgps_estimator
from configs.loader import load_config_module, require_config_attrs


USE_BAYESIAN_BGPS = os.getenv("USE_BAYESIAN_BGPS", "0").strip().lower() in {
    "1", "true", "yes", "y", "on"
}


# Column name constants for estimator results.
COLUMNS_SHARE_GTE = [
    "proposed_share_GTE", "naive_HT_share_GTE", "cluster_mean_share_GTE", "propagation_share_GTE",
    "proposed2_share_GTE", "filter_users_share_GTE", "bayesian_bgps_share_GTE", "truth_share_GTE"
]
COLUMNS_REACT_GTE = [
    "proposed_react_GTE", "proposed_react_GTE_correction", "naive_HT_react_GTE", "cluster_mean_react_GTE", "propagation_react_GTE",
    "proposed2_react_GTE", "split_credit_react_GTE", "split_credit2_react_GTE",
    "filter_users_react_GTE", "truth_react_GTE"
]
COLUMNS_SHARE_GT = [
    "proposed_share_GT", "naive_HT_share_GT", "cluster_mean_share_GT", "propagation_share_GT",
    "proposed2_share_GT", "filter_users_share_GT", "bayesian_bgps_share_GT", "truth_share_GT"
]
COLUMNS_REACT_GT = [
    "proposed_react_GT", "proposed_react_GT_correction", "naive_HT_react_GT", "cluster_mean_react_GT", "propagation_react_GT",
    "proposed2_react_GT", "split_credit_react_GT", "split_credit2_react_GT",
    "filter_users_react_GT", "truth_react_GT"
]
COLUMNS_SHARE_GC = [
    "proposed_share_GC", "naive_HT_share_GC", "cluster_mean_share_GC", "propagation_share_GC",
    "proposed2_share_GC", "filter_users_share_GC", "bayesian_bgps_share_GC", "truth_share_GC"
]
COLUMNS_REACT_GC = [
    "proposed_react_GC", "proposed_react_GC_correction", "naive_HT_react_GC", "cluster_mean_react_GC", "propagation_react_GC",
    "proposed2_react_GC", "split_credit_react_GC", "split_credit2_react_GC",
    "filter_users_react_GC", "truth_react_GC"
]
ALL_COLUMNS = (COLUMNS_SHARE_GTE + COLUMNS_REACT_GTE + 
               COLUMNS_SHARE_GT + COLUMNS_REACT_GT + 
               COLUMNS_SHARE_GC + COLUMNS_REACT_GC)


# Global worker data for multiprocessing. With fork, child processes reuse
# these objects by copy-on-write instead of serializing them.
_shared_data = None  # Will be initialized by _init_worker in worker processes


def _init_worker(G, n, m, T_max, watching_params, sharing_params,
                 pi, p, seed_base, simulation_method, inference, estimator_type,
                 cluster_baseline_labels, bayesian_verbose,
                 hawkes_ct_window_W, hawkes_ct_beta,
                 hawkes_ct_simulate_beyond_window):
    """
    Initialize shared data for worker processes.
    
    When using 'fork', this data is inherited via copy-on-write.
    When using 'spawn', this data is pickled and sent to each worker (less efficient).
    """
    global _shared_data
    _shared_data = {
        'G': G,
        'n': n,
        'm': m,
        'T_max': T_max,
        'watching_params': watching_params,
        'sharing_params': sharing_params,
        'pi': pi,
        'p': p,
        'seed_base': seed_base,
        'simulation_method': simulation_method,
        'inference': inference,
        'estimator_type': estimator_type,
        'cluster_baseline_labels': cluster_baseline_labels,
        'bayesian_verbose': bayesian_verbose,
        'hawkes_ct_window_W': hawkes_ct_window_W,
        'hawkes_ct_beta': hawkes_ct_beta,
        'hawkes_ct_simulate_beyond_window': hawkes_ct_simulate_beyond_window,
    }


def _worker_func(i):
    """
    Wrapper function for parallel execution.
    Retrieves shared data from global _shared_data and calls run_single_simulation.
    """
    return run_single_simulation(
        i,
        G=_shared_data['G'],
        n=_shared_data['n'],
        m=_shared_data['m'],
        T_max=_shared_data['T_max'],
        watching_params=_shared_data['watching_params'],
        sharing_params=_shared_data['sharing_params'],
        pi=_shared_data['pi'],
        p=_shared_data['p'],
        seed_base=_shared_data['seed_base'],
        simulation_method=_shared_data['simulation_method'],
        inference=_shared_data['inference'],
        estimator_type=_shared_data['estimator_type'],
        cluster_baseline_labels=_shared_data['cluster_baseline_labels'],
        bayesian_verbose=_shared_data['bayesian_verbose'],
        hawkes_ct_window_W=_shared_data['hawkes_ct_window_W'],
        hawkes_ct_beta=_shared_data['hawkes_ct_beta'],
        hawkes_ct_simulate_beyond_window=_shared_data['hawkes_ct_simulate_beyond_window'],
    )


def run_single_simulation(i,
                          G, n, m, T_max,
                          watching_params, sharing_params, 
                          pi, p, seed_base,
                          simulation_method='hawkes',            
                          inference = False,
                          estimator_type='ht',
                          cluster_baseline_labels=None,
                          bayesian_verbose=False,
                          hawkes_ct_window_W=0.0,
                          hawkes_ct_beta=1.0,
                          hawkes_ct_simulate_beyond_window=False,
                          verbose = False):
    """
    Run a single simulation.
    
    Parameters
    ----------
    simulation_method : str
        'discrete' or 'hawkes'
    """
    seed = seed_base + i

    # generate data based on simulation method
    if simulation_method == 'discrete':
        variables = simulate_social_sharing_discrete(
            G, n, m, T_max,
            watching_params, sharing_params, pi=pi, p=p, seed=seed,
            randomization_unit='user',
            cluster_labels=None,
            verbose=verbose
        )
    elif simulation_method == 'hawkes':
        variables = simulate_social_sharing_hawkes(
            G, n, m, T_max,
            watching_params, sharing_params, pi=pi, p=p, seed=seed,
            randomization_unit='user',
            cluster_labels=None,
            verbose=verbose
        )
    elif simulation_method == 'hawkes_ct_exp_window':
        variables = simulate_social_sharing_hawkes_ct_exp_window(
            G, n, m, T_max,
            watching_params, sharing_params,
            obs_window_W=hawkes_ct_window_W,
            beta=hawkes_ct_beta,
            pi=pi, p=p, seed=seed,
            simulate_beyond_window=hawkes_ct_simulate_beyond_window,
            randomization_unit='user',
            cluster_labels=None,
            verbose=verbose
        )
    else:
        raise ValueError(
            f"Unknown simulation_method: {simulation_method}. "
            "Use 'discrete', 'hawkes', or 'hawkes_ct_exp_window'."
        ) 
    
    # Extract variables from new structure
    V = variables['V']
    Z = variables['Z']
    X = variables['X']  # share_driven_success
    Y = variables['Y']  # indep_share_success
    R = variables['R']  # share_driven_watch_count
    RT = variables['RT']
    RC = variables['RC']
    RT_d = variables.get('RT_d', np.zeros(n, dtype=np.int64))
    RC_d = variables.get('RC_d', np.zeros(n, dtype=np.int64))
    W = variables['W']  # total watch count
    S = variables['S']  # share_success = X + Y
    S1 = variables['S1']  # share_success_with_propagation
    
    # Helper function to extract estimates from result dict
    def extract_estimates(result, prefix, has_share=True, has_react=True):
        """Extract share and react estimates for GTE/GT/GC from result dict."""
        estimates = {}
        for suffix in ['GTE', 'GT', 'GC']:
            if has_share:
                estimates[f'{prefix}_share_{suffix}'] = result[f'share_{suffix}_estimate']
            if has_react:
                estimates[f'{prefix}_react_{suffix}'] = result[f'react_{suffix}_estimate']
        return estimates
    
    # Run all estimators and collect results
    all_estimates = {}
    
    # proposed estimator
    proposed_result = proposed_estimator(n, p, pi, V, Z, Y, X, R,RT_d, RC_d,
                                         inference=inference, estimator_type=estimator_type)
    all_estimates.update(extract_estimates(proposed_result, 'proposed'))
    avar = proposed_result['share_GTE_variance']
    
    all_estimates['proposed_react_GTE_correction'] = proposed_result['react_GTE_estimate_correction']
    all_estimates['proposed_react_GT_correction'] = proposed_result['react_GT_estimate_correction']
    all_estimates['proposed_react_GC_correction'] = proposed_result['react_GC_estimate_correction']

    # naive (HT or hajek) estimator on user-side randomization
    naive_HT_result = naive_estimator_mean_with_design(
        n, p, pi, V, Z, S,
        estimator_type=estimator_type,
        randomization_unit='user',
        cluster_labels=None
    )
    all_estimates.update(extract_estimates(naive_HT_result, 'naive_HT'))

    # Optional baseline: cluster randomization + naive estimator.
    # When enabled, it replaces naive_HT_* columns with cluster-baseline values.
    if cluster_baseline_labels is not None:
        if simulation_method == 'discrete':
            variables_cluster = simulate_social_sharing_discrete(
                G, n, m, T_max,
                watching_params, sharing_params, pi=pi, p=p, seed=seed,
                randomization_unit='cluster',
                cluster_labels=cluster_baseline_labels,
                verbose=verbose
            )
        elif simulation_method == 'hawkes':
            variables_cluster = simulate_social_sharing_hawkes(
                G, n, m, T_max,
                watching_params, sharing_params, pi=pi, p=p, seed=seed,
                randomization_unit='cluster',
                cluster_labels=cluster_baseline_labels,
                verbose=verbose
            )
        elif simulation_method == 'hawkes_ct_exp_window':
            variables_cluster = simulate_social_sharing_hawkes_ct_exp_window(
                G, n, m, T_max,
                watching_params, sharing_params,
                obs_window_W=hawkes_ct_window_W,
                beta=hawkes_ct_beta,
                pi=pi, p=p, seed=seed,
                simulate_beyond_window=hawkes_ct_simulate_beyond_window,
                randomization_unit='cluster',
                cluster_labels=cluster_baseline_labels,
                verbose=verbose
            )
        else:
            raise ValueError(
                f"Unknown simulation_method: {simulation_method}. "
                "Use 'discrete', 'hawkes', or 'hawkes_ct_exp_window'."
            )

        cluster_naive_result = naive_estimator_mean_with_design(
            n, p, pi,
            variables_cluster['V'], variables_cluster['Z'], variables_cluster['S'],
            estimator_type=estimator_type,
            randomization_unit='cluster',
            cluster_labels=cluster_baseline_labels
        )
        all_estimates.update(extract_estimates(cluster_naive_result, 'cluster_mean'))
    
    # propagation estimator
    propagation_result = naive_estimator_mean_with_design(
        n, p, pi, V, Z, S1,
        estimator_type=estimator_type,
        randomization_unit='user',
        cluster_labels=None
    )
    all_estimates.update(extract_estimates(propagation_result, 'propagation'))
    
    # proposed2 estimator
    proposed2_result = proposed2_estimator(n, pi, V, Z, W, R, S, estimator_type=estimator_type)
    all_estimates.update(extract_estimates(proposed2_result, 'proposed2'))
    
    # split_credit estimator (react only)
    split_credit_result = naive_estimator_split_credit(n, pi, p, V, Z, RT, RC,
                                                       estimator_type=estimator_type)
    all_estimates.update(extract_estimates(split_credit_result, 'split_credit', has_share=False))
    
    # split_credit2 estimator (react only)
    split_credit2_result = naive_estimator_split_credit2(n, pi, p, V, Z, RT, RC,
                                                         estimator_type=estimator_type)
    all_estimates.update(extract_estimates(split_credit2_result, 'split_credit2', has_share=False))
    
    # filter_users estimator
    filter_users_result = naive_estimator_filter_users(G, V, Z, R)
    all_estimates.update(extract_estimates(filter_users_result, 'filter_users'))

    # Bayesian BGPS estimator (share-focused; uses R as outcome), controlled by env var USE_BAYESIAN_BGPS.
    if USE_BAYESIAN_BGPS:
        bayesian_result = bayesian_bgps_estimator(
            G=G,
            Z=Z,
            R=R,
            precomputed_communities=cluster_baseline_labels,
            n_samples=100,
            n_tune=500,
            n_chains=4,
            louvain_resolution=1.0,
            verbose=bayesian_verbose,
        )
    else:
        bayesian_result = {
            'share_GTE_estimate': np.nan,
            'share_GTE_variance': None,
            'share_GT_estimate': np.nan,
            'share_GC_estimate': np.nan,
            'react_GTE_estimate': np.nan,
            'react_GT_estimate': np.nan,
            'react_GC_estimate': np.nan,
            'auxiliary': None
        }
    all_estimates.update(extract_estimates(bayesian_result, 'bayesian_bgps', has_react=False))
    
    # ground truth
    truth_result = ground_truth(
        G, n, m, T_max, watching_params, sharing_params,
        simulation_method, seed,
        hawkes_ct_window_W=hawkes_ct_window_W,
        hawkes_ct_beta=hawkes_ct_beta,
        hawkes_ct_simulate_beyond_window=hawkes_ct_simulate_beyond_window,
    )
    for suffix in ['GTE', 'GT', 'GC']:
        all_estimates[f'truth_share_{suffix}'] = truth_result[f'truth_share_{suffix}']
        all_estimates[f'truth_react_{suffix}'] = truth_result[f'truth_react_{suffix}']
    
    # Build estimators list in the order matching ALL_COLUMNS
    estimators = [all_estimates.get(col, np.nan) for col in ALL_COLUMNS]
    
    results = {
        'estimators': estimators,
        'watched_by_share_num': R  # share_driven_watch_count
    }
    
    # whether make inference for the proposed estimator
    if inference:
        is_aa = check_if_aa(sharing_params)
        se = float(np.sqrt(avar)) if avar > 0 else np.nan
        alpha = 0.05 # type I error rate
        z_value = stats.norm.ppf(1-alpha/2)
        proposed_share_GTE = all_estimates['proposed_share_GTE']
        truth_share_GTE = all_estimates['truth_share_GTE']
        
        if is_aa:
            # A/A test: H0: GTE = 0
            if not np.isnan(se) and se > 0:
                t_stat = proposed_share_GTE / se
                p_value = 2 * stats.norm.sf(np.abs(t_stat))  # two-sided p-value
            else:
                t_stat = np.nan
                p_value = np.nan
            
            covered = (-z_value * se <= proposed_share_GTE <= z_value * se)
            results['inference'] = {
                'type': 'A/A',
                'GTE_truth': 0.0,
                'estimate': float(proposed_share_GTE),
                'se': se,
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'alpha': alpha,
                'is_cover_truth': bool(covered)
            }
        else:
            # A/B test: H0: GTE = truth_share_GTE
            if not np.isnan(se) and se > 0:
                t_stat = proposed_share_GTE / se
                p_value = 2 * stats.norm.sf(np.abs(t_stat))  # two-sided p-value
            else:
                t_stat = np.nan
                p_value = np.nan
            
            covered = (-z_value * se + proposed_share_GTE <= truth_share_GTE <= z_value * se + proposed_share_GTE)
            results['inference'] = {
                'type': 'A/B',
                'GTE_truth': float(truth_share_GTE),
                'estimate': float(proposed_share_GTE),
                'se': se,
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'alpha': alpha,
                'is_cover_truth': bool(covered)
            }
    
    return results


def run_simulations(n_repeats,
                    G, n, m, T_max,
                    watching_params, sharing_params, 
                    pi=1.0, p=0.5,
                    simulation_method='discrete',
                    inference = False,
                    estimator_type='ht',
                    cluster_baseline_labels=None,
                    bayesian_verbose=False,
                    hawkes_ct_window_W=0.0,
                    hawkes_ct_beta=1.0,
                    hawkes_ct_simulate_beyond_window=False,
                    seed_base=100):
    results = []

    for i in tqdm(range(n_repeats), desc="Running simulations"):
        results.append(run_single_simulation(i,
                                        G, n, m, T_max,
                                        watching_params, sharing_params, 
                                        pi, p, seed_base,
                                        simulation_method=simulation_method,
                                        estimator_type=estimator_type,
                                        inference=inference,
                                        cluster_baseline_labels=cluster_baseline_labels,
                                        bayesian_verbose=bayesian_verbose,
                                        hawkes_ct_window_W=hawkes_ct_window_W,
                                        hawkes_ct_beta=hawkes_ct_beta,
                                        hawkes_ct_simulate_beyond_window=hawkes_ct_simulate_beyond_window))

    estimator_rows = [r['estimators'] for r in results]
    row_sums_matrix = np.vstack([r['watched_by_share_num'] for r in results])  # shape = (n_repeats, n)

    df_estimators = pd.DataFrame(estimator_rows, columns=ALL_COLUMNS)

    all_results = {
        'df_estimators': df_estimators,
        'watched_by_share_num_mat': row_sums_matrix,
        'df_inference': None
    }
    if inference:
        inference_records = [r['inference'] for r in results]
        all_results['df_inference'] = pd.DataFrame(inference_records)
    return all_results


def run_simulations_parallel(n_repeats,
                             G, n, m, T_max,
                             watching_params, sharing_params, 
                             pi=1.0, p=0.5,
                             simulation_method='discrete',
                             inference = False,
                             estimator_type='ht',
                             cluster_baseline_labels=None,
                             bayesian_verbose=False,
                             hawkes_ct_window_W=0.0,
                             hawkes_ct_beta=1.0,
                             hawkes_ct_simulate_beyond_window=False,
                             seed_base=100, max_workers=None,
                             mp_start_method=None):
    """
    Run simulations in parallel.
    
    Parameters
    ----------
    mp_start_method : str or None
        Multiprocessing start method: 'fork', 'spawn', or None (system default).
        - 'fork' (Linux only): Child processes share memory with parent, more efficient
          for large read-only data like network G. Recommended for servers.
        - 'spawn' (Windows/macOS/Linux): Child processes get fresh copies of everything.
          Required for Windows. Uses more memory but safer.
        - None: Use system default (fork on Linux, spawn on Windows/macOS).
    """
    
    results = []
    
    # Determine the multiprocessing context
    if mp_start_method is not None:
        # Check if fork is requested on Windows (not supported)
        if mp_start_method == 'fork' and platform.system() == 'Windows':
            print(f"Warning: 'fork' not supported on Windows, falling back to 'spawn'")
            mp_start_method = 'spawn'
        ctx = mp.get_context(mp_start_method)
        print(f"Using multiprocessing start method: {mp_start_method}")
    else:
        ctx = mp  # Use default context
        print(f"Using default multiprocessing start method")
    
    # Use multiprocessing.Pool with shared data via initializer
    # For 'fork': data is shared via copy-on-write (efficient, no serialization)
    # For 'spawn': data is pickled and sent to workers (less efficient but works on Windows)
    with ctx.Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(G, n, m, T_max, watching_params, sharing_params,
                  pi, p, seed_base, simulation_method, inference, estimator_type,
                  cluster_baseline_labels, bayesian_verbose,
                  hawkes_ct_window_W, hawkes_ct_beta,
                  hawkes_ct_simulate_beyond_window)
    ) as pool:
        # Only pass the varying parameter 'i' to each worker
        for result in tqdm(pool.imap_unordered(_worker_func, range(n_repeats)),
                          total=n_repeats, desc="Running simulations in parallel"):
            results.append(result)

    # Extract estimators and row sums from each repeat
    estimator_rows = [r['estimators'] for r in results]
    row_sums_matrix = np.vstack([r['watched_by_share_num'] for r in results])  # shape = (n_repeats, n)

    df_estimators = pd.DataFrame(estimator_rows, columns=ALL_COLUMNS)
    
    all_results = {
        'df_estimators': df_estimators,
        'watched_by_share_num_mat': row_sums_matrix,
        'df_inference': None
    }
    if inference:
        inference_records = [r['inference'] for r in results]
        all_results['df_inference'] = pd.DataFrame(inference_records)
    return all_results


def save_and_print_results(all_results, make_inference,settings = None,
                           print_results_summary = True, print_poisson_results = True,
                           save_settings_and_results = True):
    
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
            "coverage_rate": None,
            "mean_p_value": None
        }
        inference_summary['inference_type'] = inference_type
        inference_summary["coverage_rate"] = float(df_inference['is_cover_truth'].mean())
        inference_summary["mean_p_value"] = float(df_inference['p_value'].mean())
        
        if inference_type == 'A/A':
            inference_summary["mean_GTE_truth"] = 0.0
        else:
            inference_summary["mean_GTE_truth"] = np.mean(df_inference['GTE_truth'])
            inference_summary['mean_estimate'] = np.mean(df_inference['estimate'])

        print("\n=== Inference Summary ===")
        print(f"Inference Type: {inference_summary['inference_type']}")
        print(f"Number of Repeats: {inference_summary['n_repeats']}")
        if inference_type == 'A/B':
            print(f"Mean GTE Truth: {inference_summary['mean_GTE_truth']:.6f}")
            print(f"Mean Estimate: {inference_summary['mean_estimate']:.6f}")
        else:
            print(f"Mean GTE Truth: {inference_summary['mean_GTE_truth']:.6f}")
        print(f"Coverage Rate: {inference_summary['coverage_rate']*100:.2f}%")
        print(f"Mean P-Value: {inference_summary['mean_p_value']:.6f}\n")


    # Poisson diagnostics for per-user share-driven watch counts.
    # watched_by_share_num_mat: shape = (n_repeats, n), each row is one repeat.
    alpha = 0.05
    n = watched_by_share_num_mat.shape[1]
    poisson_results = diagnose_poisson_columns(watched_by_share_num_mat, alpha=alpha)
    good_mean_var_rate = np.sum(poisson_results['good_mean_var'])/n
    avg_diff_mean_var = np.mean(poisson_results['mean_var_diff'])
    avg_ratio_mean_var = np.mean(poisson_results['mean_var_ratio'])
    valid_tests_rate = np.sum(poisson_results['valid_tests'])/n
    poisson_like_rate = np.sum(poisson_results['poisson_like'])/np.sum(poisson_results['valid_tests']) if np.sum(poisson_results['valid_tests']) >0 else np.nan
    poisson_results_summary = {
        'good_mean_var_rate': good_mean_var_rate,
        'avg_diff_mean_var': avg_diff_mean_var,
        'avg_ratio_mean_var': avg_ratio_mean_var,
        'valid_tests_rate': valid_tests_rate,
        'poisson_like_rate': poisson_like_rate
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
        

        # Save running information and results.
    save_settings_and_results = save_settings_and_results
    
    if save_settings_and_results:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("results", f"run_{ts}")
        os.makedirs(run_dir, exist_ok=True)

        settings['timestamp'] = ts
        meta_path = os.path.join(run_dir, "settings.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)

        # Save estimator and estimators summary
        estimators_csv_path = os.path.join(run_dir, 'estimators.csv')
        df_estimators.to_csv(estimators_csv_path, index_label='repeat_id')

        summary_csv_path = os.path.join(run_dir, 'estimator_summary.csv')
        df_summary.to_csv(summary_csv_path, index_label='estimator')

        # save poisson results and summary
        poisson_results_csv_path = os.path.join(run_dir, 'poisson_results.csv')
        poisson_results.to_csv(poisson_results_csv_path, index='user_id')

        poisson_results_summary_path = os.path.join(run_dir, "poisson_results_summary.json")
        with open(poisson_results_summary_path, "w", encoding="utf-8") as f:
            json.dump(poisson_results_summary, f, ensure_ascii=False, indent=2)

        # Save inference results and summary
        if make_inference:
            inference_csv_path = os.path.join(run_dir, 'inference_results.csv')
            df_inference.to_csv(inference_csv_path, index_label='repeat_id')
            
            inference_summary_path = os.path.join(run_dir, 'inference_summary.json')
            with open(inference_summary_path, "w", encoding="utf-8") as f:
                json.dump(inference_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {run_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run simulation experiments with runtime config")
    parser.add_argument(
        "--config",
        default="configs.config",
        help="Config module path only (e.g. configs.config)",
    )
    args = parser.parse_args()

    config_module, config_source = load_config_module(args.config)
    config_values = require_config_attrs(
        config_module,
        [
            "SIMULATION_METHOD", "T_MAX", "N_REPEATS", "MAX_WORKERS", "NETWORK_PARAMS",
            "POPULATION_PARAMS", "PARAMS_RELATED", "RANDOM_SEEDS", "HOMO_VALUES",
            "HETERO_RANGES", "ESTIMATOR_TYPE", "MP_START_METHOD", "CLUSTER_RANDOMIZATION",
        ],
        source_label=args.config,
    )

    SIMULATION_METHOD = config_values["SIMULATION_METHOD"]
    T_MAX = config_values["T_MAX"]
    N_REPEATS = config_values["N_REPEATS"]
    MAX_WORKERS = config_values["MAX_WORKERS"]
    NETWORK_PARAMS = config_values["NETWORK_PARAMS"]
    POPULATION_PARAMS = config_values["POPULATION_PARAMS"]
    PARAMS_RELATED = config_values["PARAMS_RELATED"]
    RANDOM_SEEDS = config_values["RANDOM_SEEDS"]
    HOMO_VALUES = config_values["HOMO_VALUES"]
    HETERO_RANGES = config_values["HETERO_RANGES"]
    ESTIMATOR_TYPE = config_values["ESTIMATOR_TYPE"]
    MP_START_METHOD = config_values["MP_START_METHOD"]
    CLUSTER_RANDOMIZATION = config_values["CLUSTER_RANDOMIZATION"]
    HAWKES_CT_WINDOW = getattr(config_module, "HAWKES_CT_WINDOW", {})

    if not isinstance(HAWKES_CT_WINDOW, dict):
        raise TypeError("HAWKES_CT_WINDOW must be a dict when provided.")

    hawkes_ct_window_W = float(HAWKES_CT_WINDOW.get("W", 0.0))
    hawkes_ct_beta = float(HAWKES_CT_WINDOW.get("beta", 1.0))
    raw_simulate_beyond_window = HAWKES_CT_WINDOW.get("simulate_beyond_window", False)
    if isinstance(raw_simulate_beyond_window, bool):
        hawkes_ct_simulate_beyond_window = raw_simulate_beyond_window
    elif isinstance(raw_simulate_beyond_window, (int, np.integer)):
        hawkes_ct_simulate_beyond_window = bool(raw_simulate_beyond_window)
    elif isinstance(raw_simulate_beyond_window, str):
        hawkes_ct_simulate_beyond_window = raw_simulate_beyond_window.strip().lower() in {
            "1", "true", "yes", "y", "on"
        }
    else:
        raise TypeError(
            "HAWKES_CT_WINDOW['simulate_beyond_window'] must be bool/int/str when provided."
        )
    
    # Use configuration values
    n = POPULATION_PARAMS['n']
    m = POPULATION_PARAMS['m']
    pi = POPULATION_PARAMS['pi']
    p = POPULATION_PARAMS['p']
    
    T_max = T_MAX
    n_repeats = N_REPEATS
    max_workers = MAX_WORKERS

    make_inference = True
    BAYESIAN_BGPS_VERBOSE = False
    
    # Generate network.
    G = generate_network(
        n,
        seed=RANDOM_SEEDS['seed_graph'],
        **NETWORK_PARAMS
    )
    n = len(G.nodes())

    # Cluster partition is computed once per graph, then reused across repeats.
    cluster_labels, cluster_metadata = prepare_cluster_randomization(
        G,
        n,
        cluster_randomization_config=CLUSTER_RANDOMIZATION
    )
    print(
        f"Cluster randomization: method={cluster_metadata['method']}, "
        f"n_clusters={cluster_metadata['n_clusters']}, "
        f"size[min/median/max]={cluster_metadata['cluster_size_min']}/"
        f"{cluster_metadata['cluster_size_median']:.1f}/"
        f"{cluster_metadata['cluster_size_max']}"
    )
    
    # Sample parameters.
    watching_params = sample_watching_params(
        n, m,
        seed=RANDOM_SEEDS['seed_params'],
        homo=PARAMS_RELATED['is_homo'],
        homo_values = HOMO_VALUES['watching'],
        hetero_ranges = HETERO_RANGES['watching']
    )
    sharing_params = sample_sharing_params(
        n, m,
        seed=RANDOM_SEEDS['seed_params'],
        homo=PARAMS_RELATED['is_homo'],
        homo_values_T=HOMO_VALUES['sharing_T'],
        homo_ds_perturbation=HOMO_VALUES['sharing_ds_perturbation'],
        homo_offsets=HOMO_VALUES['sharing_offsets'],
        hetero_ranges_T=HETERO_RANGES['sharing_T'],
        hetero_ds_perturbation=HETERO_RANGES['sharing_ds_perturbation'],
        hetero_shift_ranges=HETERO_RANGES['sharing_shift_ranges']
    )

    g_bar = 2 * G.number_of_edges() / n
    rho0 = compute_rho0_from_sharing_params(
        G=G,
        sharing_params=sharing_params,
        g_bar=g_bar
    )
    print(f"rho0 (max_k spectral radius of max-channel share matrix): {rho0:.6f}")



    # Run parallel simulations.
    start_time = time.time()
    all_results = run_simulations_parallel(
        n_repeats=n_repeats,
        G=G, n=n, m=m, T_max=T_max,
        watching_params=watching_params,
        sharing_params=sharing_params,
        pi=pi, p=p,
        simulation_method=SIMULATION_METHOD,
        inference = make_inference,
        estimator_type=ESTIMATOR_TYPE,
        cluster_baseline_labels=cluster_labels,
        bayesian_verbose=BAYESIAN_BGPS_VERBOSE,
        hawkes_ct_window_W=hawkes_ct_window_W,
        hawkes_ct_beta=hawkes_ct_beta,
        hawkes_ct_simulate_beyond_window=hawkes_ct_simulate_beyond_window,
        seed_base=RANDOM_SEEDS['seed_base'],
        max_workers=max_workers,
        mp_start_method=MP_START_METHOD
    )

    end_time = time.time()
    print(f"\nTime cost = {end_time - start_time:.3f} secs (Parallel)")



    # Save setting information.
    settings_dict = {
        "config_source": config_source,
        "simulation_method": SIMULATION_METHOD,
        "population": POPULATION_PARAMS,
        "network": NETWORK_PARAMS,
        "g_bar": g_bar,
        "rho0": rho0,
        "cluster_randomization": CLUSTER_RANDOMIZATION,
        "cluster_metadata": cluster_metadata,
        "cluster_naive_baseline_enabled": cluster_labels is not None,
        "params_related": PARAMS_RELATED,
        "simulation": {
            "T_max": T_MAX,
            "n_repeats": N_REPEATS,
            "max_workers": MAX_WORKERS,
            "make_inference": make_inference,
            "bayesian_bgps_verbose": BAYESIAN_BGPS_VERBOSE,
            "hawkes_ct_window": {
                "W": hawkes_ct_window_W,
                "beta": hawkes_ct_beta,
                "simulate_beyond_window": hawkes_ct_simulate_beyond_window,
                "T_eval": T_MAX + hawkes_ct_window_W,
            }
        },
        "random_seeds": RANDOM_SEEDS,
        "HOMO_VALUES": HOMO_VALUES,
        "HETERO_RANGES": HETERO_RANGES
    }
    
    save_and_print_results(all_results,
                           make_inference,
                           settings = settings_dict,
                           print_results_summary= True,
                           print_poisson_results= True,
                           save_settings_and_results= True)

    

    





