import argparse
import copy
import json
import os
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from configs.loader import load_config_module, require_config_attrs
from experiments import save_results_to_folder
from main import run_simulations_parallel
from utils import (
    compute_rho0_from_sharing_params,
    generate_network,
    prepare_cluster_randomization,
    sample_sharing_params,
    sample_watching_params,
)


def _format_float_tag(value, digits=4):
    """Format float for folder names, e.g. 0.1 -> 0p1, -0.3 -> m0p3."""
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _parse_targets(targets_text):
    if targets_text is None:
        return [0.05, 0.2, 0.5, 0.8]

    values = []
    for token in str(targets_text).split(","):
        token = token.strip()
        if not token:
            continue
        val = float(token)
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f"Each target rho0 must be positive and finite, got {val}.")
        values.append(val)

    if not values:
        raise ValueError("No valid rho0 targets parsed from --targets.")
    return values


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}.")
    return parsed


def _runtime_overrides_from_args(args):
    keys = ["n", "m", "n_repeats", "max_workers", "m_edges"]
    return {key: getattr(args, key) for key in keys if getattr(args, key) is not None}


def _infer_optional_random_dist_config(config_module):
    """Read optional random distribution config if present."""
    random_para_dist = getattr(config_module, "RANDOM_PARA_DIST", None)

    watching_c_by_param = getattr(config_module, "C_BY_PARAM_WATCHING", None)
    sharing_c_by_param = getattr(config_module, "C_BY_PARAM_SHARING", None)

    global_c_by_param = getattr(config_module, "C_BY_PARAM", None)
    if isinstance(global_c_by_param, dict):
        if watching_c_by_param is None and all(k in global_c_by_param for k in ("a_d", "b_d")):
            watching_c_by_param = {
                "a_d": global_c_by_param["a_d"],
                "b_d": global_c_by_param["b_d"],
            }
        if sharing_c_by_param is None and all(k in global_c_by_param for k in ("phi", "varphi", "theta")):
            sharing_c_by_param = {
                "phi": global_c_by_param["phi"],
                "varphi": global_c_by_param["varphi"],
                "theta": global_c_by_param["theta"],
            }

    return random_para_dist, watching_c_by_param, sharing_c_by_param


def _scale_sharing_params_for_prob_multiplier(sharing_params, scale):
    """Scale theta vectors so get_share_prob output is multiplied by `scale` before clipping."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be positive finite, got {scale}.")

    scaled = copy.deepcopy(sharing_params)
    for theta_key in ("theta_d", "theta_s"):
        for group in ("T", "C", "O"):
            scaled[theta_key][group] = np.asarray(scaled[theta_key][group], dtype=np.float64) * scale
    return scaled


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_rho0_scaling_experiments(
    config_path="configs.config",
    target_rho0_values=None,
    include_baseline=True,
    print_results_summary=True,
    print_poisson_results=True,
    runtime_overrides=None,
):
    config_module, config_source = load_config_module(config_path)
    config_values = require_config_attrs(
        config_module,
        [
            "SIMULATION_METHOD",
            "T_MAX",
            "N_REPEATS",
            "MAX_WORKERS",
            "NETWORK_PARAMS",
            "POPULATION_PARAMS",
            "PARAMS_RELATED",
            "RANDOM_SEEDS",
            "HOMO_VALUES",
            "HETERO_RANGES",
            "ESTIMATOR_TYPE",
            "MP_START_METHOD",
            "CLUSTER_RANDOMIZATION",
        ],
        source_label=config_path,
    )

    simulation_method = config_values["SIMULATION_METHOD"]
    t_max = config_values["T_MAX"]
    runtime_overrides = dict(runtime_overrides or {})
    n_repeats = runtime_overrides.get("n_repeats", config_values["N_REPEATS"])
    max_workers = runtime_overrides.get("max_workers", config_values["MAX_WORKERS"])
    network_params = copy.deepcopy(config_values["NETWORK_PARAMS"])
    population_params = copy.deepcopy(config_values["POPULATION_PARAMS"])
    if "n" in runtime_overrides:
        population_params["n"] = runtime_overrides["n"]
    if "m" in runtime_overrides:
        population_params["m"] = runtime_overrides["m"]
    if "m_edges" in runtime_overrides:
        network_params["m_edges"] = runtime_overrides["m_edges"]
    params_related = config_values["PARAMS_RELATED"]
    random_seeds = config_values["RANDOM_SEEDS"]
    homo_values = config_values["HOMO_VALUES"]
    hetero_ranges = config_values["HETERO_RANGES"]
    estimator_type = config_values["ESTIMATOR_TYPE"]
    mp_start_method = config_values["MP_START_METHOD"]
    cluster_randomization = config_values["CLUSTER_RANDOMIZATION"]
    hawkes_ct_window_config = getattr(config_module, "HAWKES_CT_WINDOW", {})

    if not isinstance(hawkes_ct_window_config, dict):
        raise TypeError("HAWKES_CT_WINDOW must be a dict when provided.")

    hawkes_ct_window_W = float(hawkes_ct_window_config.get("W", 0.0))
    hawkes_ct_beta = float(hawkes_ct_window_config.get("beta", 1.0))
    hawkes_ct_simulate_beyond_window = hawkes_ct_window_config.get("simulate_beyond_window", False)
    if not isinstance(hawkes_ct_simulate_beyond_window, bool):
        raise TypeError("HAWKES_CT_WINDOW['simulate_beyond_window'] must be a bool when provided.")

    make_inference = True
    bayesian_verbose = False

    n = population_params["n"]
    m = population_params["m"]
    pi = population_params["pi"]
    p = population_params["p"]

    targets = list(target_rho0_values if target_rho0_values is not None else [0.05, 0.2, 0.5, 0.8])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join("results", f"experiments_{ts}")
    os.makedirs(root_dir, exist_ok=True)

    base_config_snapshot = {
        "SIMULATION_METHOD": simulation_method,
        "T_MAX": t_max,
        "N_REPEATS": n_repeats,
        "MAX_WORKERS": max_workers,
        "NETWORK_PARAMS": network_params,
        "POPULATION_PARAMS": population_params,
        "PARAMS_RELATED": params_related,
        "RANDOM_SEEDS": random_seeds,
        "HOMO_VALUES": homo_values,
        "HETERO_RANGES": hetero_ranges,
        "ESTIMATOR_TYPE": estimator_type,
        "MP_START_METHOD": mp_start_method,
        "CLUSTER_RANDOMIZATION": cluster_randomization,
        "HAWKES_CT_WINDOW": {
            "W": hawkes_ct_window_W,
            "beta": hawkes_ct_beta,
            "simulate_beyond_window": hawkes_ct_simulate_beyond_window,
            "T_eval": t_max + hawkes_ct_window_W,
        },
    }

    print("\n[1/5] Building base graph and parameters once for all rho0 cases...")
    G = generate_network(n, seed=random_seeds["seed_graph"], **network_params)
    n = len(G.nodes())

    cluster_labels, cluster_metadata = prepare_cluster_randomization(
        G,
        n,
        cluster_randomization_config=cluster_randomization,
    )

    random_para_dist, watching_c_by_param, sharing_c_by_param = _infer_optional_random_dist_config(config_module)

    watching_params = sample_watching_params(
        n,
        m,
        seed=random_seeds["seed_params"],
        homo=params_related["is_homo"],
        homo_values=homo_values["watching"],
        hetero_ranges=hetero_ranges["watching"],
        random_para_dist=random_para_dist,
        c_by_param=watching_c_by_param,
    )

    base_sharing_params = sample_sharing_params(
        n,
        m,
        seed=random_seeds["seed_params"],
        homo=params_related["is_homo"],
        homo_values_T=homo_values["sharing_T"],
        homo_ds_perturbation=homo_values["sharing_ds_perturbation"],
        homo_offsets=homo_values["sharing_offsets"],
        hetero_ranges_T=hetero_ranges["sharing_T"],
        hetero_ds_perturbation=hetero_ranges["sharing_ds_perturbation"],
        hetero_shift_ranges=hetero_ranges["sharing_shift_ranges"],
        random_para_dist=random_para_dist,
        c_by_param=sharing_c_by_param,
    )

    g_bar = 2 * G.number_of_edges() / n
    baseline_rho0 = compute_rho0_from_sharing_params(
        G=G,
        sharing_params=base_sharing_params,
        g_bar=g_bar,
    )

    if baseline_rho0 <= 0:
        raise ValueError(
            f"Baseline rho0 must be positive for scaling, got {baseline_rho0}. "
            "Please adjust config and retry."
        )

    print(
        "[2/5] Baseline fixed. "
        f"n={n}, m={m}, edges={G.number_of_edges()}, g_bar={g_bar:.6f}, rho0_baseline={baseline_rho0:.6f}"
    )

    cases = []
    if include_baseline:
        cases.append({
            "case_type": "baseline",
            "target_rho0": baseline_rho0,
            "scale_factor": 1.0,
        })

    for target_rho0 in targets:
        scale_factor = target_rho0 / baseline_rho0
        cases.append({
            "case_type": "target",
            "target_rho0": float(target_rho0),
            "scale_factor": float(scale_factor),
        })

    exp_meta = {
        "timestamp": ts,
        "runner": "run_rho0_scaling_experiments.py",
        "config_source": config_source,
        "base_config_path": config_path,
        "layout": "fixed graph + fixed sampled params, vary rho0 scale only",
        "rho0_scaling": {
            "include_baseline": include_baseline,
            "target_rho0_values": targets,
            "baseline_rho0": float(baseline_rho0),
            "scaling_rule": "scale_factor = target_rho0 / baseline_rho0",
            "equivalence_note": "Implemented by scaling theta_d/theta_s, equivalent to multiplying get_share_prob output before clipping",
        },
        "base_parameters": {
            "simulation_method": simulation_method,
            "T_max": t_max,
            "n_repeats": n_repeats,
            "max_workers": max_workers,
            "estimator_type": estimator_type,
            "mp_start_method": mp_start_method,
            "make_inference": make_inference,
            "bayesian_bgps_verbose": bayesian_verbose,
            "random_seeds": random_seeds,
            "population": {
                "n": n,
                "m": m,
                "pi": pi,
                "p": p,
            },
            "network": network_params,
            "cluster_randomization": cluster_randomization,
            "cluster_metadata": cluster_metadata,
            "g_bar": float(g_bar),
            "hawkes_ct_window": {
                "W": hawkes_ct_window_W,
                "beta": hawkes_ct_beta,
                "simulate_beyond_window": hawkes_ct_simulate_beyond_window,
                "T_eval": t_max + hawkes_ct_window_W,
            },
            "random_para_dist": random_para_dist,
            "watching_c_by_param": watching_c_by_param,
            "sharing_c_by_param": sharing_c_by_param,
        },
    }

    _write_json(os.path.join(root_dir, "experiment_config.json"), exp_meta)
    _write_json(os.path.join(root_dir, "base_config.json"), base_config_snapshot)

    total_cases = len(cases)
    failed_cases = 0
    all_results_summary = []

    print(f"[3/5] Running {total_cases} rho0 case(s). Results root: {root_dir}")

    for idx, case in enumerate(cases, start=1):
        case_type = case["case_type"]
        target_rho0 = float(case["target_rho0"])
        scale_factor = float(case["scale_factor"])

        try:
            scaled_sharing_params = _scale_sharing_params_for_prob_multiplier(
                base_sharing_params,
                scale_factor,
            )
            actual_rho0 = compute_rho0_from_sharing_params(
                G=G,
                sharing_params=scaled_sharing_params,
                g_bar=g_bar,
            )

            if case_type == "baseline":
                subfolder_name = f"rho0_baseline_{_format_float_tag(actual_rho0)}"
            else:
                subfolder_name = (
                    f"rho0_target_{_format_float_tag(target_rho0)}"
                    f"__actual_{_format_float_tag(actual_rho0)}"
                    f"__scale_{_format_float_tag(scale_factor)}"
                )
            run_dir = os.path.join(root_dir, subfolder_name)

            print(
                f"\nCase {idx}/{total_cases}: type={case_type}, "
                f"target={target_rho0:.6f}, actual={actual_rho0:.6f}, scale={scale_factor:.6f}"
            )

            start_time = time.time()
            all_results = run_simulations_parallel(
                n_repeats=n_repeats,
                G=G,
                n=n,
                m=m,
                T_max=t_max,
                watching_params=watching_params,
                sharing_params=scaled_sharing_params,
                pi=pi,
                p=p,
                simulation_method=simulation_method,
                inference=make_inference,
                estimator_type=estimator_type,
                cluster_baseline_labels=cluster_labels,
                bayesian_verbose=bayesian_verbose,
                hawkes_ct_window_W=hawkes_ct_window_W,
                hawkes_ct_beta=hawkes_ct_beta,
                hawkes_ct_simulate_beyond_window=hawkes_ct_simulate_beyond_window,
                seed_base=random_seeds["seed_base"],
                max_workers=max_workers,
                mp_start_method=mp_start_method,
            )
            elapsed_time = time.time() - start_time

            df_summary, poisson_summary, inference_summary = save_results_to_folder(
                all_results,
                make_inference,
                run_dir,
                print_results_summary=print_results_summary,
                print_poisson_results=print_poisson_results,
            )

            case_meta = {
                "case_type": case_type,
                "target_rho0": target_rho0,
                "actual_rho0": float(actual_rho0),
                "baseline_rho0": float(baseline_rho0),
                "scale_factor": scale_factor,
                "relative_rho_error": float(abs(actual_rho0 - target_rho0) / target_rho0) if target_rho0 > 0 else None,
                "elapsed_time_sec": float(elapsed_time),
                "run_dir": run_dir,
            }
            _write_json(os.path.join(run_dir, "rho0_case_metadata.json"), case_meta)

            summary_row = {
                "case_index": idx,
                "case_type": case_type,
                "target_rho0": target_rho0,
                "actual_rho0": float(actual_rho0),
                "baseline_rho0": float(baseline_rho0),
                "scale_factor": scale_factor,
                "relative_rho_error": case_meta["relative_rho_error"],
                "elapsed_time_sec": float(elapsed_time),
                "folder": subfolder_name,
                "status": "success",
                "poisson_like_rate": poisson_summary.get("poisson_like_rate"),
                "inference_coverage_rate": inference_summary.get("coverage_rate") if inference_summary else None,
            }

            for estimator in df_summary.index:
                summary_row[f"{estimator}_mean"] = float(df_summary.loc[estimator, "mean"])
                summary_row[f"{estimator}_se"] = float(df_summary.loc[estimator, "se"])

            all_results_summary.append(summary_row)

        except Exception as exc:
            failed_cases += 1
            error_message = f"{type(exc).__name__}: {exc}"
            traceback_text = traceback.format_exc()
            print(f"ERROR in case {idx}/{total_cases}: {error_message}")
            print(traceback_text)

            all_results_summary.append({
                "case_index": idx,
                "case_type": case_type,
                "target_rho0": target_rho0,
                "baseline_rho0": float(baseline_rho0),
                "scale_factor": scale_factor,
                "status": "error",
                "error": error_message,
            })

    summary_csv_path = os.path.join(root_dir, "all_results_summary.csv")
    pd.DataFrame(all_results_summary).to_csv(summary_csv_path, index=False)

    print("\n[4/5] Sweep completed.")
    if failed_cases == 0:
        print("All rho0 cases succeeded.")
    else:
        print(f"Completed with {failed_cases} failed case(s).")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"[5/5] Done. Results saved to: {root_dir}")

    return root_dir, pd.DataFrame(all_results_summary)


def main():
    parser = argparse.ArgumentParser(
        description="Run fixed-config rho0 scaling experiments (baseline + target rho0 cases)."
    )
    parser.add_argument(
        "--config",
        default="configs.config",
        help="Config module path only (e.g. configs.config)",
    )
    parser.add_argument(
        "--targets",
        default="0.05, 0.2,0.5,0.8",
        help="Comma-separated rho0 targets, e.g. 0.05, 0.2,0.5,0.8",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="If provided, do not run baseline rho0 case.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce printed estimator/Poisson tables during each case run.",
    )
    parser.add_argument("--n", type=_positive_int, default=None, help="Override POPULATION_PARAMS['n'].")
    parser.add_argument("--m", type=_positive_int, default=None, help="Override POPULATION_PARAMS['m'].")
    parser.add_argument("--n-repeats", type=_positive_int, default=None, help="Override N_REPEATS.")
    parser.add_argument("--max-workers", type=_positive_int, default=None, help="Override MAX_WORKERS.")
    parser.add_argument("--m-edges", type=_positive_int, default=None, help="Override NETWORK_PARAMS['m_edges'].")
    args = parser.parse_args()

    targets = _parse_targets(args.targets)
    run_rho0_scaling_experiments(
        config_path=args.config,
        target_rho0_values=targets,
        include_baseline=(not args.no_baseline),
        print_results_summary=(not args.quiet),
        print_poisson_results=(not args.quiet),
        runtime_overrides=_runtime_overrides_from_args(args),
    )


if __name__ == "__main__":
    main()
