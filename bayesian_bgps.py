"""
Simplified Bayesian Estimator for Causal Effects under Network Interference
===========================================================================
Adapted from Forastiere et al. (JMLR 2022) for Bernoulli randomized experiments.

Simplifications over the original method:
- No individual propensity score stage (treatment is randomized, PS = p for all)
- No neighborhood propensity score stage (GPS is known under Bernoulli design)
- No model feedback issue (no PS parameters in outcome model)
- Three-step → one-step: only the outcome model remains

Estimand: τ_total = μ(1,1) - μ(0,0)
  where μ(z,g) = (1/|V|) Σ_{i∈V} E[Y_i(z,g)]

Model:
  Y_i ~ NegBinomial(mu_i, alpha)
  log(mu_i) = β_0 + β_z * Z_i + β_g * G_i + β_zg * Z_i * G_i + u_{C_i}
  u_{C_i} ~ N(0, σ_u^2)
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Dict, Optional


# ============================================================
# Part 1: Data Preparation
# ============================================================

def compute_neighborhood_treatment(A: sp.spmatrix, Z: np.ndarray) -> np.ndarray:
    """
    Compute G_i = proportion of treated neighbors for each node.
    
    Parameters
    ----------
    A : sparse adjacency matrix (N x N), symmetric, unweighted
    Z : treatment vector (N,), binary
    
    Returns
    -------
    G : np.ndarray
        Neighborhood treatment proportion, shape (N,).
    degree : np.ndarray
        Node degree vector, shape (N,).
    """
    degree = np.array(A.sum(axis=1)).flatten()
    treated_neighbors = np.array(A.dot(Z)).flatten()
    # Avoid division by zero for isolated nodes
    G = np.where(degree > 0, treated_neighbors / degree, 0.0)
    return G, degree


def detect_communities(A: sp.spmatrix, resolution: float = 1.0, verbose: bool = True) -> np.ndarray:
    """
    Community detection using Louvain algorithm.
    
    Parameters
    ----------
    A : sparse adjacency matrix (N x N)
    resolution : Louvain resolution parameter
    
    Returns
    -------
    labels : community labels (N,), integers from 0 to J-1
    """
    try:
        from community import community_louvain
        import networkx as nx
    except ImportError:
        raise ImportError(
            "Need python-louvain and networkx. "
            "Install: pip install python-louvain networkx"
        )
    
    # For large graphs, convert sparse matrix to networkx efficiently
    G_nx = nx.from_scipy_sparse_array(A)
    partition = community_louvain.best_partition(G_nx, resolution=resolution)
    
    N = A.shape[0]
    labels = np.array([partition[i] for i in range(N)])
    
    # Re-label to 0, 1, ..., J-1
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    
    if verbose:
        print(f"Community detection: {len(unique_labels)} communities found")
        print(f"  Community sizes: min={np.bincount(labels).min()}, "
              f"max={np.bincount(labels).max()}, "
              f"median={np.median(np.bincount(labels)):.0f}")
    
    return labels


def _normalize_community_labels(labels: np.ndarray, n: int) -> np.ndarray:
    """Validate and normalize community labels to contiguous integers [0, J-1]."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError(
            f"precomputed_communities must be a 1D array of length {n}, "
            f"got shape {labels.shape}."
        )

    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)):
            raise ValueError("precomputed_communities contains non-finite values.")
        rounded = np.rint(labels)
        if not np.allclose(labels, rounded):
            raise ValueError("precomputed_communities must contain integer-like labels.")
        labels = rounded.astype(np.int64)
    else:
        labels = labels.astype(np.int64, copy=False)

    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    normalized = np.array([label_map[val] for val in labels], dtype=np.int32)
    return normalized


# ============================================================
# Part 2: Bayesian Outcome Model (PyMC)
# ============================================================

def fit_outcome_model(
    Y: np.ndarray,
    Z: np.ndarray,
    G: np.ndarray,
    C: np.ndarray,
    n_samples: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 4,
    n_cores: Optional[int] = None,
    target_accept: float = 0.9,
    verbose: bool = True,
) -> Dict:
    """
    Fit Bayesian outcome model with community random effects.
    
    Model:
        Y_i ~ NegBinomial(mu_i, alpha)
        log(mu_i) = β_0 + β_z Z_i + β_g G_i + β_zg Z_i G_i + u_{C_i}
        
    Parameters
    ----------
    Y : outcome vector (N,), non-negative integers
    Z : treatment vector (N,), binary
    G : neighborhood treatment proportion (N,), in [0, 1]
    C : community labels (N,), integers 0 to J-1
    n_samples : number of posterior samples per chain
    n_tune : number of tuning steps
    n_chains : number of MCMC chains
    n_cores : number of parallel jobs used by PyMC sampling.
        If None, automatically uses 1 core inside daemon workers; otherwise min(n_chains, cpu_count).
    target_accept : target acceptance rate for NUTS
    verbose : whether to print diagnostics and sampling mode details
    
    Returns
    -------
    result : dict with 'trace' (arviz InferenceData) and 'model' (PyMC model)
    """
    import pymc as pm
    import arviz as az
    import multiprocessing as mp
    import os
    
    J = len(np.unique(C))
    log_y_mean = np.log(Y.mean() + 1)  # for prior calibration
    
    if verbose:
        print(f"Fitting outcome model: N={len(Y)}, J={J} communities")
        print(f"  Y: mean={Y.mean():.2f}, std={Y.std():.2f}, "
              f"min={Y.min()}, max={Y.max()}")
        print(f"  Z: mean={Z.mean():.3f}")
        print(f"  G: mean={G.mean():.3f}, std={G.std():.3f}")

    # Running PyMC's parallel chain sampling inside a daemon process causes:
    # "daemonic processes are not allowed to have children".
    # Auto-fallback to a single sampling core in that case.
    if n_cores is None:
        if mp.current_process().daemon:
            sample_cores = 1
        else:
            sample_cores = max(1, min(n_chains, os.cpu_count() or 1))
    else:
        sample_cores = max(1, int(n_cores))

    if verbose:
        if sample_cores == 1 and n_chains > 1:
            print("  Sampling mode: sequential chains (cores=1)")
        else:
            print(f"  Sampling mode: parallel chains (cores={sample_cores})")
    
    with pm.Model() as model:
        # ---- Priors ----
        # Regression coefficients: weakly informative on log-scale
        beta_0  = pm.Normal("beta_0",  mu=log_y_mean, sigma=2.0)
        beta_z  = pm.Normal("beta_z",  mu=0, sigma=1.0)
        beta_g  = pm.Normal("beta_g",  mu=0, sigma=2.0)
        beta_zg = pm.Normal("beta_zg", mu=0, sigma=1.0)
        
        # Community random effect std
        sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
        
        # Community random effects
        u = pm.Normal("u", mu=0, sigma=sigma_u, shape=J)
        
        # Overdispersion parameter for NegBinomial
        alpha = pm.Gamma("alpha", alpha=2, beta=0.5)
        
        # ---- Linear predictor (log link) ----
        log_mu = beta_0 + beta_z * Z + beta_g * G + beta_zg * Z * G + u[C]
        mu = pm.math.exp(log_mu)
        
        # ---- Likelihood ----
        Y_obs = pm.NegativeBinomial("Y_obs", mu=mu, alpha=alpha, observed=Y)
        
        # ---- MCMC ----
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            chains=n_chains,
            cores=sample_cores,
            target_accept=target_accept,
            random_seed=42,
            progressbar=verbose,
        )
    
    # Diagnostics
    if verbose:
        summary = az.summary(trace, var_names=["beta_0", "beta_z", "beta_g", 
                                                "beta_zg", "sigma_u", "alpha"])
        print("\nPosterior summary:")
        print(summary)
    
    return {"trace": trace, "model": model}


# ============================================================
# Part 3: Imputation and Estimand Computation
# ============================================================

def estimate_total_effect(
    trace,
    C: np.ndarray,
    degree: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """
    Estimate τ_total = μ(1,1) - μ(0,0) from posterior samples.
    
    For each posterior draw m:
        μ^(m)(z,g) = (1/|V|) Σ_{i∈V} exp(β_0^(m) + β_z^(m)*z + β_g^(m)*g 
                                         + β_zg^(m)*z*g + u_{C_i}^(m))
        τ^(m) = μ^(m)(1,1) - μ^(m)(0,0)
    
    Parameters
    ----------
    trace : PyMC trace (InferenceData)
    C : community labels (N,)
    degree : node degrees (N,), used to define V = {i: degree_i >= 1}
    
    Returns
    -------
    result : dict with posterior distribution and summary statistics
    """
    # Extract posterior samples, flatten chains
    beta_0  = trace.posterior["beta_0"].values.flatten()
    beta_z  = trace.posterior["beta_z"].values.flatten()
    beta_g  = trace.posterior["beta_g"].values.flatten()
    beta_zg = trace.posterior["beta_zg"].values.flatten()
    u_samples = trace.posterior["u"].values.reshape(-1, trace.posterior["u"].shape[-1])
    # u_samples shape: (M, J)
    
    M = len(beta_0)
    
    # Define V: nodes with degree >= 1
    V_mask = degree >= 1
    C_V = C[V_mask]
    N_V = V_mask.sum()
    
    if verbose:
        print(f"\nImputation: M={M} posterior draws, |V|={N_V} nodes (degree >= 1)")
    
    # For each draw, compute μ(1,1) and μ(0,0)
    tau_samples = np.empty(M)
    mu_11_samples = np.empty(M)
    mu_00_samples = np.empty(M)
    
    for m in range(M):
        u_V = u_samples[m, C_V]  # random effects for nodes in V
        
        # log(μ_i(1,1)) = β_0 + β_z*1 + β_g*1 + β_zg*1*1 + u_{C_i}
        log_mu_11 = beta_0[m] + beta_z[m] + beta_g[m] + beta_zg[m] + u_V
        
        # log(μ_i(0,0)) = β_0 + β_z*0 + β_g*0 + β_zg*0*0 + u_{C_i}
        log_mu_00 = beta_0[m] + u_V
        
        mu_11 = np.exp(log_mu_11).mean()  # average over V
        mu_00 = np.exp(log_mu_00).mean()  # average over V
        
        mu_11_samples[m] = mu_11
        mu_00_samples[m] = mu_00
        tau_samples[m] = mu_11 - mu_00
    
    # Summary
    result = {
        "tau_samples": tau_samples,
        "mu_11_samples": mu_11_samples,
        "mu_00_samples": mu_00_samples,
        "tau_mean": tau_samples.mean(),
        "tau_median": np.median(tau_samples),
        "tau_ci_95": (np.percentile(tau_samples, 2.5), 
                      np.percentile(tau_samples, 97.5)),
        "mu_11_mean": mu_11_samples.mean(),
        "mu_00_mean": mu_00_samples.mean(),
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Estimand: τ_total = μ(1,1) - μ(0,0)")
        print(f"{'='*50}")
        print(f"  μ(1,1) posterior mean: {result['mu_11_mean']:.4f}")
        print(f"  μ(0,0) posterior mean: {result['mu_00_mean']:.4f}")
        print(f"  τ_total posterior mean: {result['tau_mean']:.4f}")
        print(f"  τ_total posterior median: {result['tau_median']:.4f}")
        print(f"  τ_total 95% CI: [{result['tau_ci_95'][0]:.4f}, "
              f"{result['tau_ci_95'][1]:.4f}]")
    
    return result


# ============================================================
# Part 4: Full Pipeline
# ============================================================

def estimate_network_causal_effect(
    Y: np.ndarray,
    Z: np.ndarray,
    A: sp.spmatrix,
    G: Optional[np.ndarray] = None,
    precomputed_communities: Optional[np.ndarray] = None,
    n_samples: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 4,
    n_cores: Optional[int] = None,
    louvain_resolution: float = 1.0,
    verbose: bool = True,
) -> Dict:
    """
    Full pipeline: data prep → community detection → model fitting → imputation.
    
    Parameters
    ----------
    Y : outcome (N,), non-negative integers
    Z : treatment (N,), binary {0,1}
    A : adjacency matrix (N x N), sparse, symmetric
    G : neighborhood treatment proportion (N,), optional. 
        If None, computed from A and Z.
    precomputed_communities : optional community labels (N,).
        If provided, skips Louvain community detection and reuses these labels.
    n_samples, n_tune, n_chains, n_cores : MCMC parameters
    louvain_resolution : resolution for community detection
    
    Returns
    -------
    result : dict with all estimates and diagnostics
    """
    N = len(Y)
    if verbose:
        print(f"Network Causal Effect Estimation")
        print(f"  N = {N}, Edges = {A.nnz // 2}")
    
    # Step 1: Compute G if not provided
    if G is None:
        G, degree = compute_neighborhood_treatment(A, Z)
    else:
        degree = np.array(A.sum(axis=1)).flatten()
    
    # Step 2: Community detection (or reuse precomputed labels)
    if precomputed_communities is not None:
        C = _normalize_community_labels(precomputed_communities, N)
        counts = np.bincount(C)
        if verbose:
            print(f"Using precomputed communities: {len(counts)} communities")
            print(f"  Community sizes: min={counts.min()}, max={counts.max()}, median={np.median(counts):.0f}")
    else:
        C = detect_communities(A, resolution=louvain_resolution, verbose=verbose)
    
    # Step 3: Fit outcome model
    fit_result = fit_outcome_model(
        Y, Z, G, C,
        n_samples=n_samples,
        n_tune=n_tune,
        n_chains=n_chains,
        n_cores=n_cores,
        verbose=verbose,
    )
    
    # Step 4: Estimate total effect
    effect_result = estimate_total_effect(
        fit_result["trace"], C, degree, verbose=verbose
    )
    
    return {
        **effect_result,
        "trace": fit_result["trace"],
        "model": fit_result["model"],
        "communities": C,
        "G": G,
        "degree": degree,
    }


# ============================================================
# Part 5: Synthetic Data Generator (for validation)
# ============================================================

def generate_synthetic_data(
    N: int = 5000,
    n_communities: int = 50,
    p_treat: float = 0.5,
    within_prob: float = 0.02,
    between_prob: float = 0.002,
    beta_0: float = 2.0,
    beta_z: float = 0.5,
    beta_g: float = 1.0,
    beta_zg: float = 0.3,
    sigma_u: float = 0.3,
    alpha: float = 5.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, sp.spmatrix, Dict]:
    """
    Generate synthetic network experiment data with known ground truth.
    
    DGP:
    - Network: Stochastic Block Model
    - Treatment: Bernoulli(p_treat), independent across nodes
    - G_i: proportion of treated neighbors
    - Outcome: Y_i ~ NegBinomial(mu_i, alpha)
      log(mu_i) = β_0 + β_z Z_i + β_g G_i + β_zg Z_i G_i + u_{C_i}
      u_{C_i} ~ N(0, σ_u^2)
    
    Returns
    -------
    Y, Z, A, truth : data and ground truth dict
    """
    rng = np.random.default_rng(seed)
    
    # Community assignments
    community_size = N // n_communities
    C = np.repeat(np.arange(n_communities), community_size)
    if len(C) < N:
        C = np.concatenate([C, np.full(N - len(C), n_communities - 1)])
    
    # Generate SBM adjacency matrix
    print(f"Generating SBM network: N={N}, communities={n_communities}")
    rows, cols = [], []
    for i in range(N):
        for j in range(i + 1, min(i + 200, N)):  # limit range for speed
            prob = within_prob if C[i] == C[j] else between_prob
            if rng.random() < prob:
                rows.extend([i, j])
                cols.extend([j, i])
    
    A = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(N, N)
    )
    
    degree = np.array(A.sum(axis=1)).flatten()
    print(f"  Edges: {A.nnz // 2}, "
          f"Mean degree: {degree.mean():.1f}, "
          f"Isolated: {(degree == 0).sum()}")
    
    # Treatment: Bernoulli
    Z = rng.binomial(1, p_treat, size=N).astype(float)
    
    # Neighborhood treatment
    G, _ = compute_neighborhood_treatment(A, Z)
    
    # Random effects
    u = rng.normal(0, sigma_u, size=n_communities)
    
    # Outcome
    log_mu = beta_0 + beta_z * Z + beta_g * G + beta_zg * Z * G + u[C]
    mu = np.exp(log_mu)
    
    # NegBinomial parameterization: mean=mu, dispersion=alpha
    # scipy uses n, p parameterization: n=alpha, p=alpha/(alpha+mu)
    prob = alpha / (alpha + mu)
    Y = rng.negative_binomial(n=alpha, p=prob)
    
    # Compute ground truth μ(1,1) and μ(0,0)
    V_mask = degree >= 1
    C_V = C[V_mask]
    u_V = u[C_V]
    
    mu_11_true = np.exp(beta_0 + beta_z + beta_g + beta_zg + u_V).mean()
    mu_00_true = np.exp(beta_0 + u_V).mean()
    tau_true = mu_11_true - mu_00_true
    
    truth = {
        "beta_0": beta_0, "beta_z": beta_z, 
        "beta_g": beta_g, "beta_zg": beta_zg,
        "sigma_u": sigma_u, "alpha": alpha,
        "mu_11": mu_11_true,
        "mu_00": mu_00_true,
        "tau_total": tau_true,
        "communities_true": C,
    }
    
    print(f"\nGround truth:")
    print(f"  μ(1,1) = {mu_11_true:.4f}")
    print(f"  μ(0,0) = {mu_00_true:.4f}")
    print(f"  τ_total = μ(1,1) - μ(0,0) = {tau_true:.4f}")
    
    return Y, Z, A, truth


# ============================================================
# Part 6: Run Validation
# ============================================================

if __name__ == "__main__":
    # --- Generate synthetic data ---
    Y, Z, A, truth = generate_synthetic_data(
        N=5000,           # smaller for quick validation
        n_communities=50,
        p_treat=0.5,
        within_prob=0.02,
        between_prob=0.002,
        beta_0=2.0,
        beta_z=0.5,
        beta_g=1.0,
        beta_zg=0.3,
        sigma_u=0.3,
        alpha=5.0,
    )
    
    # --- Run estimation ---
    result = estimate_network_causal_effect(
        Y=Y, Z=Z, A=A,
        n_samples=1000,   # fewer for quick test
        n_tune=500,
        n_chains=2,
    )
    
    # --- Compare with ground truth ---
    print(f"\n{'='*50}")
    print(f"Validation Results")
    print(f"{'='*50}")
    print(f"  True τ_total:      {truth['tau_total']:.4f}")
    print(f"  Estimated τ_total: {result['tau_mean']:.4f}")
    print(f"  95% CI: [{result['tau_ci_95'][0]:.4f}, {result['tau_ci_95'][1]:.4f}]")
    
    covers = (result['tau_ci_95'][0] <= truth['tau_total'] <= result['tau_ci_95'][1])
    print(f"  CI covers truth: {covers}")
