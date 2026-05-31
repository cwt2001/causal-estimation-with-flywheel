import argparse
import copy
import json
import os
import time
from datetime import datetime

import pandas as pd

from configs.loader import load_config_module, require_config_attrs
from experiments import save_results_to_folder
from main import run_simulations_parallel
from log_capture import start_run_log_capture
from utils import (
    compute_rho0_from_sharing_params,
    generate_network,
    prepare_cluster_randomization,
    sample_sharing_params,
    sample_watching_params,
)


def _format_float_tag(value, digits=4):
    text = f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _parse_bool_like(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    raise TypeError("Boolean-like config value must be bool/int/str.")


def _parse_w_values(w_values_text):
    if w_values_text is None:
        return [0.0, 12.0, 24.0, 100.0]

    parsed = []
    for token in str(w_values_text).split(","):
        token = token.strip()
        if not token:
            continue
        val = float(token)
        if val < 0:
            raise ValueError(f"Each W must be >= 0, got {val}.")
        parsed.append(float(val))

    if not parsed:
        raise ValueError("No valid W values parsed from --w-values.")
    return parsed


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}.")
    return parsed


def _runtime_overrides_from_args(args):
    keys = ["n", "m", "n_repeats", "max_workers", "m_edges"]
    return {key: getattr(args, key) for key in keys if getattr(args, key) is not None}


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_w_window_sweep_experiments(
    config_path="configs.config",
    w_values=None,
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
    if simulation_method != "hawkes_ct_exp_window":
        print(
            "[info] SIMULATION_METHOD is not 'hawkes_ct_exp_window'. "
            "Overriding to 'hawkes_ct_exp_window' for W sweep."
        )
        simulation_method = "hawkes_ct_exp_window"

    runtime_overrides = dict(runtime_overrides or {})
    t_max = config_values["T_MAX"]
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

    hawkes_ct_window_cfg = getattr(config_module, "HAWKES_CT_WINDOW", {})
    if not isinstance(hawkes_ct_window_cfg, dict):
        raise TypeError("HAWKES_CT_WINDOW must be a dict when provided.")

    hawkes_ct_beta = float(hawkes_ct_window_cfg.get("beta", 1.0))
    hawkes_ct_simulate_beyond_window = _parse_bool_like(
        hawkes_ct_window_cfg.get("simulate_beyond_window", False),
        default=False,
    )

    make_inference = True
    bayesian_verbose = False

    n = population_params["n"]
    m = population_params["m"]
    pi = population_params["pi"]
    p = population_params["p"]

    sweep_w_values = list(w_values if w_values is not None else [0.0, 12.0, 24.0, 100.0])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join("results", f"experiments_{ts}")
    os.makedirs(root_dir, exist_ok=True)
    run_log_capture = start_run_log_capture(root_dir, prefix="slurm")

    print("\n[1/4] Building fixed graph and fixed sampled parameters...")
    G = generate_network(n, seed=random_seeds["seed_graph"], **network_params)
    n = len(G.nodes())

    cluster_labels, cluster_metadata = prepare_cluster_randomization(
        G,
        n,
        cluster_randomization_config=cluster_randomization,
    )

    watching_params = sample_watching_params(
        n,
        m,
        seed=random_seeds["seed_params"],
        homo=params_related["is_homo"],
        homo_values=homo_values["watching"],
        hetero_ranges=hetero_ranges["watching"],
    )

    sharing_params = sample_sharing_params(
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
    )

    g_bar = 2 * G.number_of_edges() / n
    rho0 = compute_rho0_from_sharing_params(
        G=G,
        sharing_params=sharing_params,
        g_bar=g_bar,
    )

    exp_meta = {
        "timestamp": ts,
        "runner": "run_obs_window_sweep_experiments.py",
        "run_logs": {
            "stdout": os.path.basename(run_log_capture.stdout_path),
            "stderr": os.path.basename(run_log_capture.stderr_path),
        },
        "config_source": config_source,
        "base_config_path": config_path,
        "layout": "fixed graph + fixed sampled params, sweep W only",
        "w_sweep": {
            "W_values": sweep_w_values,
            "setting_name_rule": "W_<tag>",
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
            "hawkes_ct_window": {
                "beta": hawkes_ct_beta,
                "simulate_beyond_window": hawkes_ct_simulate_beyond_window,
            },
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
            "rho0": float(rho0),
        },
    }

    _write_json(os.path.join(root_dir, "experiment_config.json"), exp_meta)

    print(
        f"[2/4] Start W sweep, total={len(sweep_w_values)}. "
        f"results_root={root_dir}"
    )

    summary_rows = []
    failed_cases = 0

    for idx, sweep_w in enumerate(sweep_w_values, start=1):
        setting_name = f"W_{_format_float_tag(sweep_w)}"
        run_dir = os.path.join(root_dir, setting_name)

        print(
            f"\nCase {idx}/{len(sweep_w_values)}: "
            f"setting={setting_name}, W={float(sweep_w):.6f}, beta={hawkes_ct_beta:.6f}, "
            f"simulate_beyond_window={hawkes_ct_simulate_beyond_window}"
        )

        try:
            start_time = time.time()
            all_results = run_simulations_parallel(
                n_repeats=n_repeats,
                G=G,
                n=n,
                m=m,
                T_max=t_max,
                watching_params=watching_params,
                sharing_params=sharing_params,
                pi=pi,
                p=p,
                simulation_method=simulation_method,
                inference=make_inference,
                estimator_type=estimator_type,
                cluster_baseline_labels=cluster_labels,
                bayesian_verbose=bayesian_verbose,
                hawkes_ct_window_W=float(sweep_w),
                hawkes_ct_beta=hawkes_ct_beta,
                hawkes_ct_simulate_beyond_window=hawkes_ct_simulate_beyond_window,
                seed_base=random_seeds["seed_base"],
                max_workers=max_workers,
                mp_start_method=mp_start_method,
            )
            elapsed = time.time() - start_time

            df_summary, poisson_summary, inference_summary = save_results_to_folder(
                all_results,
                make_inference,
                run_dir,
                print_results_summary=print_results_summary,
                print_poisson_results=print_poisson_results,
            )

            case_meta = {
                "setting_name": setting_name,
                "W": float(sweep_w),
                "beta": float(hawkes_ct_beta),
                "simulate_beyond_window": bool(hawkes_ct_simulate_beyond_window),
                "T_eval": float(t_max + float(sweep_w)),
                "elapsed_time_sec": float(elapsed),
                "run_dir": run_dir,
            }
            _write_json(os.path.join(run_dir, "setting_metadata.json"), case_meta)

            row = {
                "setting": setting_name,
                "W": float(sweep_w),
                "T_eval": float(t_max + float(sweep_w)),
                "beta": float(hawkes_ct_beta),
                "simulate_beyond_window": bool(hawkes_ct_simulate_beyond_window),
                "elapsed_time_sec": float(elapsed),
                "status": "success",
                "poisson_like_rate": poisson_summary.get("poisson_like_rate"),
                "inference_coverage_rate": inference_summary.get("coverage_rate") if inference_summary else None,
            }
            for estimator in df_summary.index:
                row[f"{estimator}_mean"] = float(df_summary.loc[estimator, "mean"])
                row[f"{estimator}_se"] = float(df_summary.loc[estimator, "se"])
            summary_rows.append(row)

        except Exception as exc:
            failed_cases += 1
            summary_rows.append(
                {
                    "setting": setting_name,
                    "W": float(sweep_w),
                    "T_eval": float(t_max + float(sweep_w)),
                    "beta": float(hawkes_ct_beta),
                    "simulate_beyond_window": bool(hawkes_ct_simulate_beyond_window),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[error] setting={setting_name}: {type(exc).__name__}: {exc}")

    print("\n[3/4] Writing summary files...")
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(root_dir, "all_results_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    final_meta = {
        "total_cases": len(sweep_w_values),
        "failed_cases": failed_cases,
        "success_cases": len(sweep_w_values) - failed_cases,
        "summary_csv": summary_csv,
    }
    _write_json(os.path.join(root_dir, "sweep_summary.json"), final_meta)

    print("[4/4] Finished.")
    print(f"Root folder: {root_dir}")
    print(f"Summary csv: {summary_csv}")

    run_log_capture.close()
    return root_dir, summary_csv


def main():
    parser = argparse.ArgumentParser(description="Sweep delayed observation window W for Hawkes CT simulation.")
    parser.add_argument(
        "--config",
        default="configs.config",
        help="Config module path (e.g. configs.config).",
    )
    parser.add_argument(
        "--w-values",
        default=None,
        help="Comma-separated W values, e.g. '0,12,24,100'.",
    )
    parser.add_argument(
        "--quiet-summary",
        action="store_true",
        help="Disable per-setting estimator summary print.",
    )
    parser.add_argument(
        "--quiet-poisson",
        action="store_true",
        help="Disable per-setting Poisson diagnostic print.",
    )
    parser.add_argument("--n", type=_positive_int, default=None, help="Override POPULATION_PARAMS['n'].")
    parser.add_argument("--m", type=_positive_int, default=None, help="Override POPULATION_PARAMS['m'].")
    parser.add_argument("--n-repeats", type=_positive_int, default=None, help="Override N_REPEATS.")
    parser.add_argument("--max-workers", type=_positive_int, default=None, help="Override MAX_WORKERS.")
    parser.add_argument("--m-edges", type=_positive_int, default=None, help="Override NETWORK_PARAMS['m_edges'].")

    args = parser.parse_args()
    w_values = _parse_w_values(args.w_values)

    run_w_window_sweep_experiments(
        config_path=args.config,
        w_values=w_values,
        print_results_summary=not args.quiet_summary,
        print_poisson_results=not args.quiet_poisson,
        runtime_overrides=_runtime_overrides_from_args(args),
    )


if __name__ == "__main__":
    main()
