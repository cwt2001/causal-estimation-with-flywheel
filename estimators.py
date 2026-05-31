import contextlib
import io
import warnings

import numpy as np

from utils import (
    simulate_social_sharing_discrete,
    simulate_social_sharing_hawkes,
    simulate_social_sharing_hawkes_ct_exp_window,
)
from scipy import stats


def calculate_proposed_var(n, p, pi, Y, X, R):
    """Calculate asymptotic variance for the proposed estimator.
    Args:
        n: sample size
        p: treatment probability
        pi: probability of being included in the experiment
        Y: array, indep_share_success
        X: array, share_driven_success
        R: array, share_driven_watch_count
    Returns:
        float: estimated asymptotic variance
    """
    y_bar = np.mean(Y)
    x_bar = np.mean(X)
    r_bar = np.mean(R)
    y2_bar = np.mean(Y**2)
    x2_bar = np.mean(X**2)
    r2_bar = np.mean(R**2)
    yx_bar = np.mean(Y * X)
    yr_bar = np.mean(Y * R)
    xr_bar = np.mean(X * R)
    grad = np.array([
        r_bar/(r_bar - x_bar),
        -r_bar/(r_bar - x_bar),
        y_bar*r_bar/(r_bar - x_bar)**2,
        -y_bar*r_bar/(r_bar - x_bar)**2,
        -y_bar*x_bar/(r_bar - x_bar)**2,
        y_bar*x_bar/(r_bar - x_bar)**2
    ])
    factor_TT = (1-p*pi)/(p*pi)
    factor_CC = (1 - (1-p)*pi)/((1-p)*pi)
    
    # Weight matrix (2x2): accounts for IPW sampling variance in T/C groups
    A = np.array([
        [factor_TT, -1],
        [-1, factor_CC]
    ])
    
    # Covariance matrix of U_i = (Y_i, X_i, R_i)
    B = np.array([
        [y2_bar,  yx_bar,  yr_bar],
        [yx_bar,  x2_bar,  xr_bar],
        [yr_bar,  xr_bar,  r2_bar]
    ])
    
    # S_n = A ⊗ B (Kronecker product)
    S_n = np.kron(A, B)
    sigma_n_squared = grad.T @ S_n @ grad / n

    return sigma_n_squared


def calculate_proposed_var_dim( V, Z, Y, X, R):
    """Calculate asymptotic variance for the proposed estimator (design-based).
    
    Args:
        V: experiment inclusion indicator (length-n vector)
        Z: treatment assignment (length-n vector)
        Y: array, corresponds to Y_i^d (indep_share_success)
        X: array, corresponds to Y_i^s (share_driven_success)
        R: array, corresponds to W_i^s (share_driven_watch_count)
    
    Returns:
        float: estimated asymptotic variance (se^2)
    """
    V = np.asarray(V)
    Z = np.asarray(Z)
    Y = np.asarray(Y)
    X = np.asarray(X)
    R = np.asarray(R)
    
    # Define masks for treatment and control groups
    treatment_mask = (Z == 1) & (V == 1)
    control_mask = (Z == 0) & (V == 1)
    
    n_T = np.sum(treatment_mask)
    n_C = np.sum(control_mask)
    
    # Compute sample means for each group: U_hat_a = (Y_bar_a, X_bar_a, R_bar_a)
    Y_bar_T = np.mean(Y[treatment_mask])
    X_bar_T = np.mean(X[treatment_mask])
    R_bar_T = np.mean(R[treatment_mask])
    
    Y_bar_C = np.mean(Y[control_mask])
    X_bar_C = np.mean(X[control_mask])
    R_bar_C = np.mean(R[control_mask])
    
    # Compute sample covariance matrices
    # Sigma_T = (1/(n_T-1)) * sum_{i: Z_i=1, V_i=1} (U_i - U_bar_T)(U_i - U_bar_T)^T
    U_T = np.column_stack([
        Y[treatment_mask] - Y_bar_T,
        X[treatment_mask] - X_bar_T,
        R[treatment_mask] - R_bar_T
    ])
    Sigma_T = U_T.T @ U_T / (n_T - 1)
    
    U_C = np.column_stack([
        Y[control_mask] - Y_bar_C,
        X[control_mask] - X_bar_C,
        R[control_mask] - R_bar_C
    ])
    Sigma_C = U_C.T @ U_C / (n_C - 1)
    
    # Compute gradient of g at U_hat_T and U_hat_C
    # g(u1, u2, u3) = u1 / (1 - u2/u3) = u1 * u3 / (u3 - u2)
    # grad g = [u3/(u3-u2), u1*u3/(u3-u2)^2, -u1*u2/(u3-u2)^2]
    d_T = R_bar_T - X_bar_T
    grad_g_T = np.array([
        R_bar_T / d_T,
        Y_bar_T * R_bar_T / (d_T ** 2),
        -Y_bar_T * X_bar_T / (d_T ** 2)
    ])
    
    d_C = R_bar_C - X_bar_C
    grad_g_C = np.array([
        R_bar_C / d_C,
        Y_bar_C * R_bar_C / (d_C ** 2),
        -Y_bar_C * X_bar_C / (d_C ** 2)
    ])
    
    # Compute se^2 = n_T^{-1} * grad_g_T^T @ Sigma_T @ grad_g_T 
    #              + n_C^{-1} * grad_g_C^T @ Sigma_C @ grad_g_C
    var_T = (grad_g_T.T @ Sigma_T @ grad_g_T) / n_T
    var_C = (grad_g_C.T @ Sigma_C @ grad_g_C) / n_C
    
    se_squared = var_T + var_C
    
    return se_squared


def proposed_estimator(n, p, pi, V, Z,
                        Y, X, R, RT_d, RC_d,
                        inference=False, estimator_type='ht'):
    """
    Proposed estimator for share/react GTE.
    
    Args:
        n, p, pi: population parameters
        V: experiment inclusion indicator
        Z: treatment assignment
        indep_share_success, share_driven_success, share_driven_watch_count: outcome variables
        inference: whether to compute variance
        estimator_type: 'ht' (Horvitz-Thompson) or 'hajek'
            - 'ht': uses n*p*pi and n*(1-p)*pi as denominators
            - 'hajek': uses actual treatment/control sample counts as denominators
    """
    V = np.asarray(V)
    Z = np.asarray(Z)

    treatment_mask = (Z == 1) & (V == 1)
    control_mask = (Z == 0) & (V == 1)

    # Choose denominator based on estimator type
    if estimator_type == 'hajek':
        n_T = np.sum(treatment_mask)  # actual treatment sample size
        n_C = np.sum(control_mask)    # actual control sample size
        denom_T = n_T
        denom_C = n_C
    else:  # 'ht' (default)
        denom_T = n * p * pi
        denom_C = n * (1 - p) * pi

    hat_theta_T = np.sum(Y[treatment_mask]) / denom_T if denom_T > 0 else np.nan
    hat_theta_C = np.sum(Y[control_mask]) / denom_C if denom_C > 0 else np.nan
    hat_q_T = np.sum(X[treatment_mask]) / np.sum(R[treatment_mask])
    hat_q_C = np.sum(X[control_mask]) / np.sum(R[control_mask])

    hat_share_GT = hat_theta_T / (1 - hat_q_T)
    hat_share_GC = hat_theta_C / (1 - hat_q_C)
    hat_share_GTE = hat_share_GT - hat_share_GC

    hat_react_GT = 1 - np.exp(-hat_share_GT)
    hat_react_GC = 1 - np.exp(-hat_share_GC)
    hat_react_GTE = np.exp(-hat_share_GT) * np.expm1(hat_share_GT - hat_share_GC)

    # Estimate variance of lambda for reactivation corrections.
    lam_T_variance_hat = np.maximum(np.var(RT_d,ddof=1) - np.mean(RT_d),0)
    lam_T_variance_hat = lam_T_variance_hat/((np.sum(treatment_mask)/n)**2)
    lam_T_variance_hat = lam_T_variance_hat / ((1- hat_q_T)**2)
    lam_C_variance_hat = np.maximum(np.var(RC_d,ddof=1) - np.mean(RC_d),0)
    lam_C_variance_hat = lam_C_variance_hat/((np.sum(control_mask)/n)**2)
    lam_C_variance_hat = lam_C_variance_hat / ((1- hat_q_C)**2)

    # gamma correction using estimated variance of lambda
    if lam_T_variance_hat > 0:
        hat_react_GT_correction = 1 - (1+lam_T_variance_hat/hat_share_GT)**(-hat_share_GT**2/lam_T_variance_hat)
    else:
        hat_react_GT_correction = 1 - np.exp(-hat_share_GT)
    if lam_C_variance_hat > 0:
        hat_react_GC_correction = 1 - (1+lam_C_variance_hat/hat_share_GC)**(-hat_share_GC**2/lam_C_variance_hat)
    else:
        hat_react_GC_correction = 1 - np.exp(-hat_share_GC)
    hat_react_GTE_correction = hat_react_GT_correction - hat_react_GC_correction

    if inference:
        hat_avar_share_GTE = calculate_proposed_var_dim(V, Z,Y, X, R)
    else:
        hat_avar_share_GTE = None

    return {
        'share_GTE_estimate': hat_share_GTE,
        'share_GTE_variance': hat_avar_share_GTE,
        'share_GT_estimate': hat_share_GT,
        'share_GC_estimate': hat_share_GC,
        'react_GTE_estimate': hat_react_GTE,
        'react_GT_estimate': hat_react_GT,
        'react_GC_estimate': hat_react_GC,
        'react_GTE_estimate_correction': hat_react_GTE_correction,
        'react_GT_estimate_correction': hat_react_GT_correction,
        'react_GC_estimate_correction': hat_react_GC_correction,
        'auxiliary': {
            'theta_T': hat_theta_T,
            'theta_C': hat_theta_C,
            'q_T': hat_q_T,
            'q_C': hat_q_C
        }
    }

def proposed2_estimator(n, pi, V, Z, W, R, S, estimator_type='ht'):
    """
    Proposed estimator variant 2.
    
    Args:
        n, pi: population parameters
        V: experiment inclusion indicator
        Z: treatment assignment
        W: total watch count per user (length-n vector)
        R: share-driven watch count per user (length-n vector)
        S: share success count per user (length-n vector)
        estimator_type: 'ht' (Horvitz-Thompson) or 'hajek'
    """
    V = np.asarray(V)
    Z = np.asarray(Z)
    W = np.asarray(W)
    R = np.asarray(R)
    S = np.asarray(S)
    
    mask_T = (Z == 1) & (V == 1)
    mask_C = (Z == 0) & (V == 1)
    
    # Choose denominator based on estimator type
    if estimator_type == 'hajek':
        n_exp = np.sum(V == 1)  # actual experiment sample size
        denom_exp = n_exp
    else:  # 'ht' (default)
        denom_exp = n * pi

    # Independent watch count = total watch - share-driven watch
    indep_watch = W - R
    theta_hat = np.sum(indep_watch[V==1]) / denom_exp if denom_exp > 0 else np.nan

    # p_hat = share_success / total_watch for each group
    p_T_hat = np.sum(S[mask_T]) / np.sum(W[mask_T]) if np.sum(W[mask_T]) > 0 else np.nan
    p_C_hat = np.sum(S[mask_C]) / np.sum(W[mask_C]) if np.sum(W[mask_C]) > 0 else np.nan

    hat_share_GT = theta_hat * p_T_hat / (1 - p_T_hat)
    hat_share_GC = theta_hat * p_C_hat / (1 - p_C_hat)

    hat_share_GTE = hat_share_GT - hat_share_GC

    hat_react_GT = 1 - np.exp(-hat_share_GT)
    hat_react_GC = 1 - np.exp(-hat_share_GC)
    hat_react_GTE = np.exp(-hat_share_GT) * np.expm1(hat_share_GT - hat_share_GC)

    return {
        'share_GTE_estimate': hat_share_GTE,
        'share_GTE_variance': None,
        'share_GT_estimate': hat_share_GT,
        'share_GC_estimate': hat_share_GC,
        'react_GTE_estimate': hat_react_GTE,
        'react_GT_estimate': hat_react_GT,
        'react_GC_estimate': hat_react_GC,
        'auxiliary': {
            'p_T_hat': p_T_hat,
            'p_C_hat': p_C_hat,
            'theta_hat': theta_hat
        }
    }


def naive_estimator_split_credit(n, pi, p, V, Z, RT, RC, assign_credit='average', estimator_type='ht'):
    '''
    Naive estimator for reactivation GTE based on successful shares
    
    Args:
        n, pi, p: population parameters
        V: experiment inclusion indicator
        Z: treatment assignment
        RT: share-driven watch count from Treatment senders (length-n vector)
        RC: share-driven watch count from Control senders (length-n vector)
        assign_credit: credit assignment method
        estimator_type: 'ht' (Horvitz-Thompson) or 'hajek'
    '''
    V = np.asarray(V)
    Z = np.asarray(Z)
    RT = np.asarray(RT)
    RC = np.asarray(RC)
    treatment_mask = (Z == 1) & (V == 1)
    control_mask = (Z == 0) & (V == 1)
    if estimator_type == 'hajek':
        n_T = np.sum(treatment_mask)
        n_C = np.sum(control_mask)
        weight_T = n_T / n
        weight_C = n_C / n
    else:  # 'ht' (default)
        assert pi * p > 0 and pi * (1 - p) > 0
        weight_T = pi * p
        weight_C = pi * (1 - p)
    react_T = 0
    react_C = 0
    if assign_credit == 'average':
        for j in range(n):
            if V[j] == 0:
                if RT[j] + RC[j] > 0:
                    tmp_react_T_j = RT[j] / weight_T if weight_T > 0 else np.nan
                    tmp_react_C_j = RC[j] / weight_C if weight_C > 0 else np.nan
                    react_T += tmp_react_T_j / (tmp_react_T_j + tmp_react_C_j)
                    react_C += tmp_react_C_j / (tmp_react_T_j + tmp_react_C_j)
    else:
        raise NotImplementedError("Only 'average' credit assignment is implemented")
    if estimator_type == 'hajek':
        n_outside = np.sum(V == 0)
        denom = n_outside
    else:
        denom = (1 - pi) * n
    hat_react_GT = react_T / denom if denom > 0 else np.nan 
    hat_react_GC = react_C / denom if denom > 0 else np.nan
    hat_react_GTE = hat_react_GT - hat_react_GC
    return {
        'react_GTE_estimate': hat_react_GTE,
        'react_GTE_variance': None,
        'react_GT_estimate': hat_react_GT,
        'react_GC_estimate': hat_react_GC,
        'auxiliary': {
            'react_T': react_T,
            'react_C': react_C
        }
    }


def naive_estimator_split_credit2(n, pi, p, V, Z, RT, RC, assign_credit='average', estimator_type='ht'):
    '''
    Naive estimator for reactivation GTE based on successful shares (variant 2)
    
    Args:
        n, pi, p: population parameters
        V: experiment inclusion indicator
        Z: treatment assignment
        RT: share-driven watch count from Treatment senders (length-n vector)
        RC: share-driven watch count from Control senders (length-n vector)
        assign_credit: credit assignment method
        estimator_type: 'ht' (Horvitz-Thompson) or 'hajek'
    '''
    V = np.asarray(V)
    Z = np.asarray(Z)
    RT = np.asarray(RT)
    RC = np.asarray(RC)
    
    treatment_mask = (Z == 1) & (V == 1)
    control_mask = (Z == 0) & (V == 1)
    
    # For internal weight calculation in split credit
    if estimator_type == 'hajek':
        n_T = np.sum(treatment_mask)
        n_C = np.sum(control_mask)
        weight_T = n_T/n 
        weight_C = n_C/n 
    else:  # 'ht' (default)
        assert pi * p > 0 and pi * (1 - p) > 0
        weight_T = pi * p
        weight_C = pi * (1 - p)
    
    react_T = 0
    react_C = 0
    if assign_credit == 'average':
        for j in range(n):
            if RT[j] + RC[j] > 0:
                tmp_react_T_j = RT[j] / weight_T if weight_T > 0 else np.nan
                tmp_react_C_j = RC[j] / weight_C if weight_C > 0 else np.nan
                react_T += tmp_react_T_j / (tmp_react_T_j + tmp_react_C_j)
                react_C += tmp_react_C_j / (tmp_react_T_j + tmp_react_C_j)
    else:
        raise NotImplementedError("Only 'average' credit assignment is implemented")
    hat_react_GT = react_T / n 
    hat_react_GC = react_C / n
    hat_react_GTE = hat_react_GT - hat_react_GC

    return {
        'react_GTE_estimate': hat_react_GTE,
        'react_GTE_variance': None,
        'react_GT_estimate': hat_react_GT,
        'react_GC_estimate': hat_react_GC,
        'auxiliary': {
            'react_T': react_T,
            'react_C': react_C
        }
    }


def naive_estimator_filter_users(G, V, Z, R):
    """
    Naive estimator for reactivation GTE by filtering users outside the experiment
    whose friends are all in treatment or all in control.
    
    Args:
        G: NetworkX graph
        V: array indicating experiment inclusion (1=in experiment, 0=outside)
        Z: array indicating treatment assignment (1=treatment, 0=control)
        R: share-driven watch count per user (length-n vector)
    
    Returns:
        dict with react_estimate and auxiliary information
    """
    V = np.asarray(V)
    Z = np.asarray(Z)
    R = np.asarray(R)
    n = len(V)
    
    # Find users outside experiment whose friends are all treatment or all control
    pure_T_users = []  # Users outside experiment with all friends in treatment
    pure_C_users = []  # Users outside experiment with all friends in control
    
    for j in range(n):
        if V[j] == 1:  # Skip users in experiment
            continue
        
        neighbors = list(G.neighbors(j))
        if not neighbors:  # Skip users with no friends
            continue
        
        # Check neighbors in experiment
        neighbors_in_exp = [i for i in neighbors if V[i] == 1]
        if len(neighbors_in_exp) == 0:  # No friends in experiment
            continue
        
        # Check treatment status of neighbors in experiment
        treatment_neighbors = [i for i in neighbors_in_exp if Z[i] == 1]
        control_neighbors = [i for i in neighbors_in_exp if Z[i] == 0]
        
        if len(treatment_neighbors) == len(neighbors_in_exp):
            # All friends in experiment are in treatment
            pure_T_users.append(j)
        elif len(control_neighbors) == len(neighbors_in_exp):
            # All friends in experiment are in control
            pure_C_users.append(j)
    
    # Calculate average shares received
    # For pure_T_users: all shares come from T group, so R[j] = shares from T
    # For pure_C_users: all shares come from C group, so R[j] = shares from C
    if len(pure_T_users) > 0:
        hat_share_GT = np.mean(R[pure_T_users])
        hat_react_GT = np.mean((R[pure_T_users] >= 1).astype(int))
    else:
        hat_share_GT = np.nan
        hat_react_GT = np.nan 
    
    if len(pure_C_users) > 0:
        hat_share_GC = np.mean(R[pure_C_users])
        hat_react_GC = np.mean((R[pure_C_users] >= 1).astype(int))
    else:
        hat_share_GC = np.nan
        hat_react_GC = np.nan
    
    hat_share_GTE = hat_share_GT - hat_share_GC
    hat_react_GTE = hat_react_GT - hat_react_GC
    
    return {
        'share_GTE_estimate': hat_share_GTE,
        'share_GTE_variance': None,
        'share_GT_estimate': hat_share_GT,
        'share_GC_estimate': hat_share_GC,
        'react_GTE_estimate': hat_react_GTE,
        'react_GT_estimate': hat_react_GT,
        'react_GC_estimate': hat_react_GC,
        'auxiliary': {
            'n_pure_T_users': len(pure_T_users),
            'n_pure_C_users': len(pure_C_users)
        }
    }


def bayesian_bgps_estimator(G, Z, R,
                            precomputed_communities=None,
                            n_samples=1000,
                            n_tune=500,
                            n_chains=4,
                            louvain_resolution=1.0,
                            verbose=False):
    """Bayesian BGPS wrapper aligned with the project estimator interface.

    Uses R (share-driven watch count) as the outcome and returns tau_mean as share_GTE.
    If Bayesian fitting fails for a repeat, NaNs are returned so the experiment can continue.
    """
    try:
        Z = np.asarray(Z, dtype=np.int8)
        R = np.asarray(R)

        if R.ndim != 1 or Z.ndim != 1 or R.shape[0] != Z.shape[0]:
            raise ValueError("R and Z must be 1D arrays with the same length.")

        if not np.all(np.isfinite(R)):
            raise ValueError("R contains non-finite values.")

        # Bayesian BGPS expects non-negative integer outcomes.
        Y_outcome = R.astype(np.int64)

        import networkx as nx
        from bayesian_bgps import estimate_network_causal_effect

        A = nx.to_scipy_sparse_array(G, format='csr')
        if verbose:
            bgps_result = estimate_network_causal_effect(
                Y=Y_outcome,
                Z=Z,
                A=A,
                precomputed_communities=precomputed_communities,
                n_samples=n_samples,
                n_tune=n_tune,
                n_chains=n_chains,
                louvain_resolution=louvain_resolution,
                verbose=True,
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bgps_result = estimate_network_causal_effect(
                    Y=Y_outcome,
                    Z=Z,
                    A=A,
                    precomputed_communities=precomputed_communities,
                    n_samples=n_samples,
                    n_tune=n_tune,
                    n_chains=n_chains,
                    louvain_resolution=louvain_resolution,
                    verbose=False,
                )

        share_gte = float(bgps_result.get('tau_mean', np.nan))
        share_gt = float(bgps_result.get('mu_11_mean', np.nan))
        share_gc = float(bgps_result.get('mu_00_mean', np.nan))
        aux = {
            'status': 'ok',
            'tau_ci_95': bgps_result.get('tau_ci_95'),
            'n_communities': int(len(np.unique(bgps_result.get('communities')))),
        }
    except Exception as exc:
        if verbose:
            print(f"Error in Bayesian BGPS estimator: {exc}")
        share_gte = np.nan
        share_gt = np.nan
        share_gc = np.nan
        aux = {
            'status': 'error',
            'error': str(exc),
        }

    return {
        'share_GTE_estimate': share_gte,
        'share_GTE_variance': None,
        'share_GT_estimate': share_gt,
        'share_GC_estimate': share_gc,
        'react_GTE_estimate': np.nan,
        'react_GT_estimate': np.nan,
        'react_GC_estimate': np.nan,
        'auxiliary': aux,
    }


def naive_estimator_mean(n, p, pi, V, Z, sender_y, estimator_type='ht'):
    """
    Mean estimator for share/react GTE.
    
    Args:
        n, p, pi: population parameters
        V: experiment inclusion indicator
        Z: treatment assignment
        sender_y: outcome variable
        estimator_type: 'ht' (Horvitz-Thompson) or 'hajek'
            - 'ht': uses n*p*pi and n*(1-p)*pi as denominators
            - 'hajek': uses actual treatment/control sample counts as denominators
    """
    return naive_estimator_mean_with_design(
        n=n,
        p=p,
        pi=pi,
        V=V,
        Z=Z,
        sender_y=sender_y,
        estimator_type=estimator_type,
        randomization_unit='user',
        cluster_labels=None
    )


def naive_estimator_mean_with_design(n, p, pi, V, Z, sender_y,
                                     estimator_type='ht',
                                     randomization_unit='user',
                                     cluster_labels=None):
    """Mean estimator with user-level or cluster-level randomization design.

    For randomization_unit='cluster':
      - Let m1 be the number of treated clusters, m0 the number of control clusters.
      - First compute per-cluster mean of target quantity.
      - Then average those cluster means across treated/control clusters.
    """
    V = np.asarray(V)
    Z = np.asarray(Z)
    sender_y = np.asarray(sender_y)

    if randomization_unit == 'cluster':
        if cluster_labels is None:
            raise ValueError("cluster_labels is required when randomization_unit='cluster'.")

        cluster_labels = np.asarray(cluster_labels)
        if cluster_labels.shape[0] != n:
            raise ValueError(
                f"cluster_labels length ({cluster_labels.shape[0]}) must match n ({n})."
            )

        unique_clusters = np.unique(cluster_labels)
        m_total = len(unique_clusters)

        cluster_means = []
        cluster_V = []
        cluster_Z = []

        for cid in unique_clusters:
            idx = (cluster_labels == cid)
            if not np.any(idx):
                continue

            v_unique = np.unique(V[idx])
            z_unique = np.unique(Z[idx])
            if v_unique.size != 1 or z_unique.size != 1:
                raise ValueError(
                    "In cluster randomization mode, all units in a cluster must share the same V and Z."
                )

            cluster_means.append(float(np.mean(sender_y[idx])))
            cluster_V.append(int(v_unique[0]))
            cluster_Z.append(int(z_unique[0]))

        cluster_means = np.asarray(cluster_means, dtype=float)
        cluster_V = np.asarray(cluster_V, dtype=int)
        cluster_Z = np.asarray(cluster_Z, dtype=int)

        treated_cluster_mask = (cluster_V == 1) & (cluster_Z == 1)
        control_cluster_mask = (cluster_V == 1) & (cluster_Z == 0)

        m1 = int(np.sum(treated_cluster_mask))
        m0 = int(np.sum(control_cluster_mask))

        if estimator_type == 'hajek':
            denom_T = m1
            denom_C = m0
        else:  # 'ht'
            denom_T = m_total * p * pi
            denom_C = m_total * (1 - p) * pi

        num_T = np.sum(cluster_means[treated_cluster_mask]) if m1 > 0 else 0.0
        num_C = np.sum(cluster_means[control_cluster_mask]) if m0 > 0 else 0.0

        hat_share_GT = num_T / denom_T if denom_T > 0 else np.nan
        hat_share_GC = num_C / denom_C if denom_C > 0 else np.nan
    else:
        treatment_mask = (Z == 1) & (V == 1)
        control_mask = (Z == 0) & (V == 1)
    
        # Choose denominator based on estimator type
        if estimator_type == 'hajek':
            n_T = np.sum(treatment_mask)  # actual treatment sample size
            n_C = np.sum(control_mask)    # actual control sample size
            denom_T = n_T
            denom_C = n_C
        else:  # 'ht' (default)
            denom_T = n * p * pi
            denom_C = n * (1 - p) * pi

        hat_share_GT = np.sum(sender_y * V * Z) / denom_T if denom_T > 0 else np.nan
        hat_share_GC = np.sum(sender_y * V * (1 - Z)) / denom_C if denom_C > 0 else np.nan

    hat_share_GTE = hat_share_GT - hat_share_GC

    hat_react_GT = 1 - np.exp(-hat_share_GT)
    hat_react_GC = 1 - np.exp(-hat_share_GC)
    hat_react_GTE = np.exp(-hat_share_GT) * np.expm1(hat_share_GT - hat_share_GC)

    return {
        'share_GTE_estimate': hat_share_GTE,
        'share_GTE_variance': None,
        'share_GT_estimate': hat_share_GT,
        'share_GC_estimate': hat_share_GC,
        'react_GTE_estimate': hat_react_GTE,
        'react_GT_estimate': hat_react_GT,
        'react_GC_estimate': hat_react_GC,
        'auxiliary': {
            'design': randomization_unit
        }
    }


def ground_truth(G, n, m, T_max,
                 watching_params, sharing_params,
                 simulation_method,
                 seed,
                 hawkes_ct_window_W=0.0,
                 hawkes_ct_beta=1.0,
                 hawkes_ct_simulate_beyond_window=False,
                 ):
    if simulation_method == 'discrete':
        variables_T = simulate_social_sharing_discrete(
            G, n, m, T_max,
            watching_params, sharing_params, 
            pi=1.0, p=1.0, seed=seed
        )
        variables_C = simulate_social_sharing_discrete(
            G, n, m, T_max,
            watching_params, sharing_params, 
            pi=1.0, p=0.0, seed=seed
        )
    elif simulation_method == 'hawkes':
        variables_T = simulate_social_sharing_hawkes(
            G, n, m, T_max,
            watching_params, sharing_params, 
            pi=1.0, p=1.0, seed=seed
        )
        variables_C = simulate_social_sharing_hawkes(
            G, n, m, T_max,
            watching_params, sharing_params, 
            pi=1.0, p=0.0, seed=seed
        )
    elif simulation_method == 'hawkes_ct_exp_window':
        variables_T = simulate_social_sharing_hawkes_ct_exp_window(
            G, n, m, T_max,
            watching_params, sharing_params,
            obs_window_W=hawkes_ct_window_W,
            beta=hawkes_ct_beta,
            simulate_beyond_window=hawkes_ct_simulate_beyond_window,
            pi=1.0, p=1.0, seed=seed
        )
        variables_C = simulate_social_sharing_hawkes_ct_exp_window(
            G, n, m, T_max,
            watching_params, sharing_params,
            obs_window_W=hawkes_ct_window_W,
            beta=hawkes_ct_beta,
            simulate_beyond_window=hawkes_ct_simulate_beyond_window,
            pi=1.0, p=0.0, seed=seed
        )
    else:
        raise ValueError(
            "Unknown simulation method: {}. Use 'discrete', 'hawkes', or "
            "'hawkes_ct_exp_window'.".format(simulation_method)
        )
    
    # R is share_driven_watch_count (total shares received per user)
    R_T = variables_T['R']
    R_C = variables_C['R']

    # Share GTE (expected number of shares received per user)
    truth_share_GT = np.mean(R_T)
    truth_share_GC = np.mean(R_C)
    truth_share_GTE = truth_share_GT - truth_share_GC

    # React GTE (probability of being reactivated)
    truth_react_GT = np.mean((R_T >= 1).astype(int))
    truth_react_GC = np.mean((R_C >= 1).astype(int))
    truth_react_GTE = truth_react_GT - truth_react_GC

    return {
        'truth_share_GTE': truth_share_GTE,
        'truth_share_GT': truth_share_GT,
        'truth_share_GC': truth_share_GC,
        'truth_react_GTE': truth_react_GTE,
        'truth_react_GT': truth_react_GT,
        'truth_react_GC': truth_react_GC,
        'auxiliary': {}
    }


# calculate true GTE by theoretical formula
def ground_truth_theor(G, n, m, watching_params, sharing_params):

    g_bar = 2*G.number_of_edges() / n  # average degree
    # Note: applicable to the special parameter structure
    # Precompute terms
    i_terms_d_T = watching_params['a_d'] * sharing_params["phi_d"]['T'] 
    i_terms_d_C = watching_params['a_d'] * sharing_params["phi_d"]['C'] 
    i_terms_s_T = sharing_params["phi_s"]['T'] 
    i_terms_s_C = sharing_params["phi_s"]['C'] 
    
    j_terms_d_T = sharing_params["varphi_d"]['T'] 
    j_terms_d_C = sharing_params["varphi_d"]['C'] 
    j_terms_s_T = sharing_params["varphi_s"]['T'] 
    j_terms_s_C = sharing_params["varphi_s"]['C'] 
    
    k_terms_d_T = watching_params['b_d'] * sharing_params["theta_d"]['T'] 
    k_terms_d_C = watching_params['b_d'] * sharing_params["theta_d"]['C'] 
    k_terms_s_T = sharing_params["theta_s"]['T'] 
    k_terms_s_C = sharing_params["theta_s"]['C'] 

    k_terms_d_T_sum = np.sum(k_terms_d_T)
    k_terms_d_C_sum = np.sum(k_terms_d_C)
    k_terms_s_T_sum = np.sum(k_terms_s_T)
    k_terms_s_C_sum = np.sum(k_terms_s_C)

    # Initialize results
    theta_T = 0.0
    theta_C = 0.0
    q_T = 0.0
    q_C = 0.0

    # Compute theta and q
    for i in range(n):
        neighbors = list(G.neighbors(i))
        if not neighbors:
            continue
        
        # Sum of neighbor j terms
        neighbor_sum_d_T = np.sum(j_terms_d_T[neighbors])
        neighbor_sum_d_C = np.sum(j_terms_d_C[neighbors])
        neighbor_sum_s_T = np.sum(j_terms_s_T[neighbors])
        neighbor_sum_s_C = np.sum(j_terms_s_C[neighbors])

        # Compute theta (global contribution)
        theta_T += i_terms_d_T[i] * neighbor_sum_d_T * k_terms_d_T_sum
        theta_C += i_terms_d_C[i] * neighbor_sum_d_C * k_terms_d_C_sum
        
        q_T += i_terms_s_T[i] * neighbor_sum_s_T * k_terms_s_T_sum
        q_C += i_terms_s_C[i] * neighbor_sum_s_C * k_terms_s_C_sum

    # Normalize theta
    theta_T /= (n *  m* g_bar)
    theta_C /= (n *  m* g_bar)

    q_T /= (n *  m* g_bar)
    q_C /= (n *  m* g_bar)

     # Compute denominator
    denom_T = 1.0 - q_T
    denom_C = 1.0 - q_C
    
    # Share GTE (theoretical)
    truth_theor_share_GT = theta_T / denom_T
    truth_theor_share_GC = theta_C / denom_C
    truth_theor_share_GTE = truth_theor_share_GT - truth_theor_share_GC

    # React GTE (theoretical)
    truth_theor_react_GT = 1 - np.exp(-truth_theor_share_GT)
    truth_theor_react_GC = 1 - np.exp(-truth_theor_share_GC)
    truth_theor_react_GTE = np.exp(-truth_theor_share_GT) * np.expm1(truth_theor_share_GT - truth_theor_share_GC)

    return {
        'truth_theor_share_GTE': truth_theor_share_GTE,
        'truth_theor_share_GT': truth_theor_share_GT,
        'truth_theor_share_GC': truth_theor_share_GC,
        'truth_theor_react_GTE': truth_theor_react_GTE,
        'truth_theor_react_GT': truth_theor_react_GT,
        'truth_theor_react_GC': truth_theor_react_GC,
        'auxiliary': {
            'theta_T': theta_T,
            'theta_C': theta_C,
            'q_T': q_T,
            'q_C': q_C
        }
    }

