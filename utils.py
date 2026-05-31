import time
import heapq
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, deque
from scipy import sparse, stats


def get_memory_usage(variables):
    """Calculate memory usage of variables.
    
    Args:
        variables: dict containing various data structures
    
    Returns:
        dict: memory usage per variable (in MB)
    """
    import sys
    
    memory_usage = defaultdict(float)
    
    def get_sparse_matrix_size(matrix):
        if not sparse.issparse(matrix):
            return 0
        return (matrix.data.nbytes + matrix.indices.nbytes + 
                matrix.indptr.nbytes) / (1024 * 1024)
    
    def get_array_size(arr):
        return arr.nbytes / (1024 * 1024)
    
    def get_watch_log_size(watch_log):
        if watch_log is None:
            return 0
        total_size = sys.getsizeof(watch_log)
        for record in watch_log:
            total_size += sys.getsizeof(record)
            for item in record:
                total_size += sys.getsizeof(item)
        return total_size / (1024 * 1024)
    
    # Calculate memory usage for each variable
    memory_usage['has_watched'] = get_sparse_matrix_size(variables['has_watched'])
    memory_usage['has_watched_by_share'] = get_sparse_matrix_size(variables['has_watched_by_share'])
    memory_usage['V'] = get_array_size(variables['V'])
    memory_usage['Z'] = get_array_size(variables['Z'])
    

    memory_usage['watch_log'] = get_watch_log_size(variables['watch_log'])
    
    memory_usage['total'] = sum(memory_usage.values())
    
    return dict(memory_usage)


def check_if_aa(sharing_params):
    """Check if Control and Treatment groups have identical sharing parameters."""
    param_names = ['phi_d', 'phi_s', 'varphi_d', 'varphi_s', 'theta_d', 'theta_s']
    
    for param_name in param_names:
        param_dict = sharing_params.get(param_name, {})
        val_T = param_dict.get('T')
        val_C = param_dict.get('C')
        
        val_T = np.asarray(val_T)
        val_C = np.asarray(val_C)
        
        if not np.allclose(val_T, val_C):
            return False
    
    return True


def _sample_from_random_para_dist(dist_name, c_value, size):
    """Sample non-negative values with moment-matched distributions parameterized by c.

    Target moments are matched to U(0, c):
    E[X] = c/2, Var[X] = c^2/12.
    """
    if dist_name is None:
        raise ValueError("dist_name cannot be None when using random_para_dist sampling")

    dist_name = str(dist_name).strip().lower()
    c_value = float(c_value)

    if not np.isfinite(c_value):
        raise ValueError(f"c must be finite, got {c_value}")
    if c_value < 0:
        raise ValueError(f"c must be non-negative, got {c_value}")
    if c_value == 0:
        return np.zeros(size, dtype=float)

    if dist_name == 'uniform':
        return np.random.uniform(0.0, c_value, size=size)

    if dist_name == 'lognormal':
        # Match moments to U(0, c): sigma^2 = ln(4/3), mu = ln(c/2) - sigma^2/2.
        sigma2 = np.log(4.0 / 3.0)
        sigma = np.sqrt(sigma2)
        mu = np.log(c_value / 2.0) - 0.5 * sigma2
        return np.random.lognormal(mean=mu, sigma=sigma, size=size)

    if dist_name == 'gamma':
        # Match moments to U(0, c): shape=3, scale=c/6.
        return np.random.gamma(shape=3.0, scale=c_value / 6.0, size=size)

    raise ValueError(
        f"Unsupported random_para_dist '{dist_name}'. "
        "Supported values are: uniform, lognormal, gamma."
    )


def sample_watching_params(n, m, seed=123, homo=False, homo_values=None, hetero_ranges=None,
                          random_para_dist=None, c_by_param=None):
    np.random.seed(seed)
    
    if homo:
        values = homo_values
        a_d = np.array([values['a_d']] * n)
        b_d = np.array([values['b_d']] * m)
    else:
        if random_para_dist is None:
            ranges = hetero_ranges
            a_d = np.random.uniform(*ranges['a_d'], size=n)
            b_d = np.random.uniform(*ranges['b_d'], size=m)
        else:
            if c_by_param is None:
                raise ValueError("c_by_param is required when random_para_dist is provided")

            missing_keys = [key for key in ('a_d', 'b_d') if key not in c_by_param]
            if missing_keys:
                raise ValueError(
                    f"Missing keys in c_by_param for watching params: {missing_keys}"
                )

            a_d = _sample_from_random_para_dist(random_para_dist, c_by_param['a_d'], size=n)
            b_d = _sample_from_random_para_dist(random_para_dist, c_by_param['b_d'], size=m)

    watching_params = {
        'a_d': a_d,
        'b_d': b_d
    }
    return watching_params

     

def sample_sharing_params(n, m, seed=123, homo=False, 
                          homo_values_T=None, homo_ds_perturbation=None, homo_offsets=None,
                          hetero_ranges_T=None, hetero_ds_perturbation=None, hetero_shift_ranges=None,
                          random_para_dist=None, c_by_param=None):
    """
    Sample sharing parameters for users and videos.
    
    New structure:
    - _d params are base values
    - _s params = _d params + perturbation (d and s are correlated)
    - C/O groups = T group + shift (same shift applied to both _d and _s)
    """
    np.random.seed(seed)

    sharing_params = {
        "phi_d": {},      # phi_i^{d,·}
        "phi_s": {},      # phi_i^{s,·}
        "varphi_d": {},   # varphi_j^{d,·}
        "varphi_s": {},   # varphi_j^{s,·}
        "theta_d": {},    # theta_k^{d,·}
        "theta_s": {}     # theta_k^{s,·}
    }

    if homo:
        values = homo_values_T
        ds_perturb = homo_ds_perturbation
        
        # Base values for _d
        sharing_params["phi_d"]['T'] = np.array([values['phi']] * n)
        sharing_params["varphi_d"]['T'] = np.array([values['varphi']] * n)
        sharing_params["theta_d"]['T'] = np.array([values['theta']] * m)
        
        # _s = _d + perturbation
        sharing_params["phi_s"]['T'] = np.maximum(sharing_params["phi_d"]['T'] + ds_perturb['phi'], 0)
        sharing_params["varphi_s"]['T'] = np.maximum(sharing_params["varphi_d"]['T'] + ds_perturb['varphi'], 0)
        sharing_params["theta_s"]['T'] = np.maximum(sharing_params["theta_d"]['T'] + ds_perturb['theta'], 0)
        
        # C and O groups: T + offset (same offset for _d and _s)
        offsets = homo_offsets
        for group in ["C", "O"]:
            shift_phi = offsets['phi'][group]
            shift_varphi = offsets['varphi'][group]
            shift_theta = offsets['theta'][group]
            
            sharing_params["phi_d"][group] = np.maximum(sharing_params["phi_d"]['T'] + shift_phi, 0)
            sharing_params["phi_s"][group] = np.maximum(sharing_params["phi_s"]['T'] + shift_phi, 0)
            
            sharing_params["varphi_d"][group] = np.maximum(sharing_params["varphi_d"]['T'] + shift_varphi, 0)
            sharing_params["varphi_s"][group] = np.maximum(sharing_params["varphi_s"]['T'] + shift_varphi, 0)
            
            sharing_params["theta_d"][group] = np.maximum(sharing_params["theta_d"]['T'] + shift_theta, 0)
            sharing_params["theta_s"][group] = np.maximum(sharing_params["theta_s"]['T'] + shift_theta, 0)
    else:
        ranges = hetero_ranges_T
        ds_perturb = hetero_ds_perturbation
        
        # Base values for _d (legacy: uniform ranges; optional: random_para_dist with c_by_param)
        if random_para_dist is None:
            sharing_params["phi_d"]['T'] = np.random.uniform(*ranges['phi'], size=n)
            sharing_params["varphi_d"]['T'] = np.random.uniform(*ranges['varphi'], size=n)
            sharing_params["theta_d"]['T'] = np.random.uniform(*ranges['theta'], size=m)
        else:
            if c_by_param is None:
                raise ValueError("c_by_param is required when random_para_dist is provided")

            missing_keys = [key for key in ('phi', 'varphi', 'theta') if key not in c_by_param]
            if missing_keys:
                raise ValueError(
                    f"Missing keys in c_by_param for sharing params: {missing_keys}"
                )

            sharing_params["phi_d"]['T'] = _sample_from_random_para_dist(
                random_para_dist, c_by_param['phi'], size=n
            )
            sharing_params["varphi_d"]['T'] = _sample_from_random_para_dist(
                random_para_dist, c_by_param['varphi'], size=n
            )
            sharing_params["theta_d"]['T'] = _sample_from_random_para_dist(
                random_para_dist, c_by_param['theta'], size=m
            )
        
        # _s = _d + perturbation (from uniform range)
        sharing_params["phi_s"]['T'] = np.maximum(
            sharing_params["phi_d"]['T'] + np.random.uniform(*ds_perturb['phi'], size=n), 0)
        sharing_params["varphi_s"]['T'] = np.maximum(
            sharing_params["varphi_d"]['T'] + np.random.uniform(*ds_perturb['varphi'], size=n), 0)
        sharing_params["theta_s"]['T'] = np.maximum(
            sharing_params["theta_d"]['T'] + np.random.uniform(*ds_perturb['theta'], size=m), 0)
        
        # C and O groups: T + shift (same shift for _d and _s)
        shift_ranges = hetero_shift_ranges
        for group in ["C", "O"]:
            # Sample shift once, apply to both _d and _s
            shift_phi = np.random.uniform(*shift_ranges['phi'][group], size=n)
            shift_varphi = np.random.uniform(*shift_ranges['varphi'][group], size=n)
            shift_theta = np.random.uniform(*shift_ranges['theta'][group], size=m)
            
            sharing_params["phi_d"][group] = np.maximum(sharing_params["phi_d"]['T'] + shift_phi, 0)
            sharing_params["phi_s"][group] = np.maximum(sharing_params["phi_s"]['T'] + shift_phi, 0)
            
            sharing_params["varphi_d"][group] = np.maximum(sharing_params["varphi_d"]['T'] + shift_varphi, 0)
            sharing_params["varphi_s"][group] = np.maximum(sharing_params["varphi_s"]['T'] + shift_varphi, 0)
            
            sharing_params["theta_d"][group] = np.maximum(sharing_params["theta_d"]['T'] + shift_theta, 0)
            sharing_params["theta_s"][group] = np.maximum(sharing_params["theta_s"]['T'] + shift_theta, 0)

    return sharing_params


def generate_network(n, method='barabasi', **kwargs):
    seed = kwargs.pop('seed', None)

    method = (method or 'barabasi').lower()

    if method == 'barabasi':
        G = nx.barabasi_albert_graph(n, int(kwargs.get('m_edges', 6) / 2), seed=seed)
    elif method == 'watts':
        G = nx.watts_strogatz_graph(n, kwargs.get('m_edges', 6), kwargs.get('graph_p', 0.1), seed=seed)
    elif method == 'regular':
        k = kwargs.get('m_edges', 2)
        if k * n % 2 != 0:
            raise ValueError(f"Cannot create regular graph: degree ({k}) * n ({n}) must be even")
        if k >= n:
            raise ValueError(f"Degree ({k}) must be less than n ({n})")
        G = nx.random_regular_graph(k, n, seed=seed)
    elif method == 'abc':
        # Special 3-node graph with edges (0-1) and (0-2), retained for tests.
        if n != 3:
            raise ValueError("For method='abc', n must be 3 to match the special 3-node graph.")
        G = nx.Graph()
        G.add_nodes_from(range(3))
        G.add_edges_from([(0, 1), (0, 2)])
    elif method in ('sbm', 'stochastic_block_model'):
        K = int(kwargs.get('sbm_k', kwargs.get('K', 1)))
        sbm_probs = kwargs.get('sbm_probs', None)

        if K < 1:
            raise ValueError(f"sbm_k must be >= 1, got {K}")
        if n is None or n <= 0:
            raise ValueError(f"n must be a positive integer for SBM, got {n}")
        if K > n:
            raise ValueError(f"sbm_k ({K}) cannot exceed n ({n})")

        base_size = n // K
        remainder = n % K
        sizes = [base_size + (1 if i < remainder else 0) for i in range(K)]

        if sbm_probs is not None:
            probs = np.asarray(sbm_probs, dtype=float)
            if probs.shape != (K, K):
                raise ValueError(f"sbm_probs must have shape ({K}, {K}), got {probs.shape}")
            if np.any((probs < 0.0) | (probs > 1.0)):
                raise ValueError("All entries in sbm_probs must be in [0, 1]")
            # For undirected SBM, enforce a symmetric probability matrix.
            if not np.allclose(probs, probs.T):
                raise ValueError("sbm_probs must be symmetric for undirected graphs")
            probs = probs.tolist()
        else:
            d_val = kwargs.get('sbm_d', kwargs.get('d', None))
            eta_val = kwargs.get('sbm_eta', kwargs.get('eta', None))

            if d_val is not None and eta_val is not None:
                d_val = float(d_val)
                eta_val = float(eta_val)
                if d_val < 0.0:
                    raise ValueError(f"sbm_d must be >= 0, got {d_val}")
                if eta_val < 0.0:
                    raise ValueError(f"sbm_eta must be >= 0, got {eta_val}")

                denom = (base_size - 1) + (n - base_size) * eta_val
                if denom <= 0.0:
                    raise ValueError(
                        "Invalid SBM degree parameterization: "
                        f"(base_size - 1) + (n - base_size) * sbm_eta must be > 0, got {denom}"
                    )

                p_in = d_val / denom
                p_out = eta_val * p_in
            else:
                # Backward-compatible fallback when sbm_d/sbm_eta are not provided.
                p_in = float(kwargs.get('sbm_p_in', kwargs.get('p_in', kwargs.get('graph_p', 0.1))))
                p_out = float(kwargs.get('sbm_p_out', kwargs.get('p_out', p_in)))

            if not (0.0 <= p_in <= 1.0):
                raise ValueError(
                    f"Derived sbm_p_in must be in [0, 1], got {p_in}. "
                    "Adjust sbm_d/sbm_eta (or sbm_p_in)."
                )
            if not (0.0 <= p_out <= 1.0):
                raise ValueError(
                    f"Derived sbm_p_out must be in [0, 1], got {p_out}. "
                    "Adjust sbm_d/sbm_eta (or sbm_p_out)."
                )

            probs = [[p_in if i == j else p_out for j in range(K)] for i in range(K)]

        G = nx.stochastic_block_model(sizes, probs, seed=seed)
    elif method in ('er', 'erdos_renyi'):
        p_er = float(kwargs.get('er_p', 0.001))
        if not (0.0 <= p_er <= 1.0):
            raise ValueError(f"er_p must be in [0, 1], got {p_er}")
        G = nx.erdos_renyi_graph(n, p_er, seed=seed)
    elif method == 'fbsnd':
        G = nx.read_edgelist('data/facebook_combined.txt')
        mapping = {old: int(old) for old in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        mapping = {node: idx for idx, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping, copy=False)
    elif method == 'brightkite':
        G = nx.read_edgelist('data/Brightkite_edges.txt')
        mapping = {old: int(old) for old in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        mapping = {node: idx for idx, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping, copy=False)
    elif method == 'twitch_gamer':
        import csv
        path = "data/large_twitch_edges.csv"
        G = nx.Graph()
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            G.add_edges_from((int(r["numeric_id_1"]), int(r["numeric_id_2"])) for r in reader)
        mapping = {node: idx for idx, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping, copy=False)
    else:
        raise ValueError("Unsupported method")
    return G


def _normalize_partition_labels(partition_dict):
    """Relabel cluster IDs to consecutive integers [0, K-1]."""
    unique_labels = sorted(set(partition_dict.values()))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    return {node: label_mapping[label] for node, label in partition_dict.items()}


def detect_graph_clusters(G, method='louvain', resolution=1.0, seed=None):
    """Detect communities on graph G and return {node: cluster_id}."""
    method = (method or 'louvain').lower()

    if method == 'louvain':
        partition = None
        try:
            # Preferred when available: python-louvain package
            from community import community_louvain
            partition = community_louvain.best_partition(
                G,
                resolution=resolution,
                random_state=seed
            )
        except ImportError:
            # Fallback: NetworkX built-in Louvain
            if not hasattr(nx.community, 'louvain_communities'):
                raise ImportError(
                    "Louvain clustering requires either python-louvain package "
                    "or networkx with community.louvain_communities support."
                )
            communities = nx.community.louvain_communities(G, resolution=resolution, seed=seed)
            partition = {}
            for cid, nodes in enumerate(communities):
                for node in nodes:
                    partition[node] = cid

        return _normalize_partition_labels(partition)

    if method == 'leiden':
        try:
            import igraph as ig
            import leidenalg as la
        except ImportError as exc:
            raise ImportError(
                "Leiden clustering requires 'igraph' and 'leidenalg'. "
                "Install via: pip install igraph leidenalg"
            ) from exc

        nodes = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

        g_ig = ig.Graph(n=len(nodes), edges=edge_list, directed=False)
        partition_obj = la.find_partition(
            g_ig,
            la.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=seed
        )

        partition = {}
        for cid, member_indices in enumerate(partition_obj):
            for idx in member_indices:
                partition[nodes[idx]] = cid

        return _normalize_partition_labels(partition)

    raise ValueError(f"Unsupported cluster method: {method}. Use 'louvain' or 'leiden'.")


def prepare_cluster_randomization(G, n, cluster_randomization_config=None):
    """Pre-compute cluster labels once per graph for cluster randomization."""
    default_cfg = {
        'method': 'louvain',      # 'louvain' or 'leiden'
        'resolution': 1.0,
        'seed': None
    }
    cfg = default_cfg.copy()
    if cluster_randomization_config is not None:
        cfg.update(cluster_randomization_config)

    method = (cfg.get('method') or 'louvain').lower()
    resolution = float(cfg.get('resolution', 1.0))
    seed = cfg.get('seed', None)

    partition = detect_graph_clusters(G, method=method, resolution=resolution, seed=seed)
    cluster_labels = np.full(n, -1, dtype=np.int32)

    for node, cid in partition.items():
        node_id = int(node)
        if 0 <= node_id < n:
            cluster_labels[node_id] = int(cid)

    if np.any(cluster_labels < 0):
        missing = int(np.sum(cluster_labels < 0))
        raise ValueError(
            f"Cluster labeling failed: {missing} users are unlabeled. "
            "Expected node IDs to match [0, n-1]."
        )

    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    metadata = {
        'unit': 'cluster',
        'method': method,
        'resolution': resolution,
        'seed': seed,
        'n_clusters': int(len(unique_clusters)),
        'cluster_size_min': int(np.min(counts)),
        'cluster_size_median': float(np.median(counts)),
        'cluster_size_max': int(np.max(counts))
    }
    return cluster_labels, metadata


def assign_treatment(n, pi=0.8, p=0.5, seed=None,
                     randomization_unit='user', cluster_labels=None):
    if seed is not None:
        np.random.seed(seed)

    unit = (randomization_unit or 'user').lower()

    if unit == 'user':
        V = np.random.binomial(1, pi, size=n).astype(np.int8)
        Z = np.zeros(n, dtype=np.int8)
        Z[V == 1] = np.random.binomial(1, p, size=(V == 1).sum()).astype(np.int8)
        return V, Z

    if unit != 'cluster':
        raise ValueError(f"Unsupported randomization unit: {unit}. Use 'user' or 'cluster'.")

    if cluster_labels is None:
        raise ValueError("cluster_labels is required when randomization_unit='cluster'.")

    cluster_labels = np.asarray(cluster_labels)
    if cluster_labels.shape[0] != n:
        raise ValueError(
            f"cluster_labels length ({cluster_labels.shape[0]}) must match n ({n})."
        )

    unique_clusters, inverse = np.unique(cluster_labels, return_inverse=True)

    V_cluster = np.random.binomial(1, pi, size=len(unique_clusters)).astype(np.int8)
    Z_cluster = np.zeros(len(unique_clusters), dtype=np.int8)
    eligible_cluster_mask = (V_cluster == 1)
    Z_cluster[eligible_cluster_mask] = np.random.binomial(
        1, p, size=int(np.sum(eligible_cluster_mask))
    ).astype(np.int8)

    V = V_cluster[inverse]
    Z = Z_cluster[inverse]
    return V, Z


def get_group(Vi, Zi):
    if Vi == 0:
        return 'O'
    elif Zi == 1:
        return 'T'
    else:
        return 'C'


def get_share_prob(g_bar, i, j, k, sharing_params, group, mode="d"):
    if mode == "d":
        phi = sharing_params["phi_d"][group][i]
        varphi = sharing_params["varphi_d"][group][j]
        theta = sharing_params["theta_d"][group][k]
    elif mode == 's':
        phi = sharing_params["phi_s"][group][i]
        varphi = sharing_params["varphi_s"][group][j]
        theta = sharing_params["theta_s"][group][k]
    else:
        raise ValueError(f"Unknown mode {mode}.")
    return np.clip(phi * varphi * theta / g_bar, 0, 1) # restrict to [0, 1]


def _spectral_radius_sparse_matrix(matrix, tol=1e-8, max_iter=2000):
    """Compute spectral radius of a sparse non-negative matrix."""
    if matrix.nnz == 0:
        return 0.0

    n = matrix.shape[0]
    if n == 1:
        return float(np.abs(matrix[0, 0]))

    try:
        eigvals = sparse.linalg.eigs(
            matrix.astype(np.float64),
            k=1,
            which='LM',
            return_eigenvectors=False
        )
        return float(np.abs(eigvals[0]))
    except Exception:
        # Fallback avoids dense conversion for large n when ARPACK does not converge.
        x = np.full(n, 1.0 / n, dtype=np.float64)
        rho_prev = 0.0

        for _ in range(max_iter):
            y = matrix @ x
            y_norm1 = np.linalg.norm(y, ord=1)
            if y_norm1 == 0.0:
                return 0.0

            x = y / y_norm1
            Ax = matrix @ x
            denom = float(np.dot(x, x))
            rho = float(np.abs(np.dot(x, Ax) / denom)) if denom > 0.0 else 0.0

            if np.abs(rho - rho_prev) <= tol * max(1.0, rho):
                return rho
            rho_prev = rho

        return rho_prev


def _coerce_sharing_params_arrays(sharing_params):
    """Validate sharing_params and coerce all parameter vectors to float arrays."""
    required_param_keys = ('phi_d', 'phi_s', 'varphi_d', 'varphi_s', 'theta_d', 'theta_s')
    required_groups = ('T', 'C', 'O')

    validated = {}
    for param_key in required_param_keys:
        if param_key not in sharing_params:
            raise ValueError(f"Missing sharing_params['{param_key}'].")

        param_by_group = sharing_params[param_key]
        validated[param_key] = {}

        for group in required_groups:
            if group not in param_by_group:
                raise ValueError(f"Missing sharing_params['{param_key}']['{group}'].")

            arr = np.asarray(param_by_group[group], dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(
                    f"sharing_params['{param_key}']['{group}'] must be 1-D, got shape {arr.shape}."
                )
            validated[param_key][group] = arr

    n = validated['phi_d']['T'].shape[0]
    m = validated['theta_d']['T'].shape[0]

    for param_key in ('phi_d', 'phi_s', 'varphi_d', 'varphi_s'):
        for group in required_groups:
            if validated[param_key][group].shape[0] != n:
                raise ValueError(
                    f"Inconsistent user dimension in sharing_params['{param_key}']['{group}']: "
                    f"expected {n}, got {validated[param_key][group].shape[0]}."
                )

    for param_key in ('theta_d', 'theta_s'):
        for group in required_groups:
            if validated[param_key][group].shape[0] != m:
                raise ValueError(
                    f"Inconsistent item dimension in sharing_params['{param_key}']['{group}']: "
                    f"expected {m}, got {validated[param_key][group].shape[0]}."
                )

    return validated, n, m


def compute_rho0_from_sharing_params(G, sharing_params, g_bar):
    """Compute rho0 = max_k rho(M_k) for edge-only share-probability matrices.

    For each item k, define M_k with:
      M_k[i, j] = max over 6 channels
                  {(mode, group) in {d, s} x {T, C, O}}
                  of clip(phi_i * varphi_j * theta_k / g_bar, 0, 1),
                  only on graph edges (non-edges are 0).

    Returns:
      float: rho0 = max_k spectral_radius(M_k)
    """
    if G is None:
        raise ValueError("G cannot be None.")
    if g_bar is None or (not np.isfinite(g_bar)) or g_bar <= 0:
        raise ValueError(f"g_bar must be a positive finite number, got {g_bar}.")

    validated, n, m = _coerce_sharing_params_arrays(sharing_params)

    try:
        edges = np.asarray(list(G.edges()), dtype=np.int64)
    except Exception as exc:
        raise ValueError("Graph node IDs must be integer-castable for matrix indexing.") from exc

    if edges.size == 0:
        return 0.0
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("G.edges() must provide edge pairs (u, v).")

    if G.is_directed():
        rows = edges[:, 0]
        cols = edges[:, 1]
    else:
        rows = np.concatenate((edges[:, 0], edges[:, 1]))
        cols = np.concatenate((edges[:, 1], edges[:, 0]))

    if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= n) or np.any(cols >= n):
        raise ValueError(
            "Graph node IDs must lie in [0, n-1], where n is implied by sharing_params user dimension."
        )

    channel_specs = [
        ('phi_d', 'varphi_d', 'theta_d', 'T'),
        ('phi_s', 'varphi_s', 'theta_s', 'T'),
        ('phi_d', 'varphi_d', 'theta_d', 'C'),
        ('phi_s', 'varphi_s', 'theta_s', 'C'),
        ('phi_d', 'varphi_d', 'theta_d', 'O'),
        ('phi_s', 'varphi_s', 'theta_s', 'O'),
    ]

    edge_bases = []
    theta_vectors = []
    for phi_key, varphi_key, theta_key, group in channel_specs:
        phi_arr = validated[phi_key][group]
        varphi_arr = validated[varphi_key][group]
        theta_arr = validated[theta_key][group]

        edge_bases.append((phi_arr[rows] * varphi_arr[cols]) / g_bar)
        theta_vectors.append(theta_arr)

    rho0 = 0.0
    weights = np.zeros(rows.shape[0], dtype=np.float64)

    for k in range(m):
        weights.fill(0.0)

        for edge_base, theta_arr in zip(edge_bases, theta_vectors):
            channel_probs = np.clip(edge_base * theta_arr[k], 0.0, 1.0)
            np.maximum(weights, channel_probs, out=weights)

        Mk = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
        rho_k = _spectral_radius_sparse_matrix(Mk)
        if rho_k > rho0:
            rho0 = rho_k

    return float(rho0)


def simulate_social_sharing_discrete(G, n, m, T_max, watching_params,
                            sharing_params, 
                            pi=1.0, p=0.5, seed=None,
                            randomization_unit='user', cluster_labels=None,
                            verbose=False):
    """
    Memory-optimized discrete-time simulation for social sharing.
    
    Returns aggregated statistics instead of storing full watch history.
    
    Returns:
        dict with keys:
            V: experiment inclusion indicator
            Z: treatment assignment
            X: share-driven success count (user shared after receiving a share)
            Y: independent success count (user shared after independent watch)
            R: total share-driven watch count per user
            RT: share-driven watch from Treatment group senders
            RC: share-driven watch from Control group senders
            W: total watch count per user
            S: total share success count (S = X + Y)
            S1: share success with 1-step propagation credit
    """
    if seed is not None:
        np.random.seed(seed)
    
    g_bar = 2*G.number_of_edges() / n 

    V, Z = assign_treatment(
        n, pi, p, seed=seed,
        randomization_unit=randomization_unit,
        cluster_labels=cluster_labels
    )

    a_d = watching_params['a_d']
    b_d = watching_params['b_d']

    # Aggregated statistics (length-n vectors)
    X = np.zeros(n, dtype=np.int64)  # share success after share-driven watch
    Y = np.zeros(n, dtype=np.int64)  # share success after independent watch
    R = np.zeros(n, dtype=np.int64)  # total share-driven watch count
    RT = np.zeros(n, dtype=np.int64) # share-driven from Treatment
    RC = np.zeros(n, dtype=np.int64) # share-driven from Control
    W = np.zeros(n, dtype=np.int64)  # total watch count
    S1_credit = np.zeros(n, dtype=np.int64)  # 1-step propagation credit

    # Current time step events: list of (user, item, generation, parent_user)
    current_events = []

    # Time 1: Independent watching
    b_max = np.max(b_d)
    for i in range(n):
        p_max_i = (a_d[i] * b_max) / m
        num_candidates = np.random.binomial(m, p_max_i)
        if num_candidates > 0:
            candidates = np.random.choice(m, size=num_candidates, replace=False)
            accept_mask = np.random.rand(num_candidates) < (b_d[candidates] / b_max)
            selected = candidates[accept_mask]
            W[i] += len(selected)
            for k in selected:
                current_events.append((i, int(k), 0, -1))

    # Time ≥ 2: Shared watching
    for t in range(2, T_max + 1):
        next_events = []
        new_share_num = 0
        
        for (i, k, v, parent) in current_events:
            mode = 'd' if v == 0 else 's'
            group = get_group(V[i], Z[i])
            
            for j in G.neighbors(i):
                p_share = get_share_prob(g_bar, i, j, k, sharing_params, group, mode=mode)
                if np.random.rand() < p_share:
                    new_share_num += 1
                    
                    # Update statistics for receiver j
                    W[j] += 1
                    R[j] += 1
                    if V[i] == 1:
                        if Z[i] == 1:
                            RT[j] += 1
                        else:
                            RC[j] += 1
                    
                    # Update statistics for sender i
                    if v == 0:
                        Y[i] += 1
                    else:
                        X[i] += 1
                    
                    # Update 1-step propagation credit
                    if v >= 1 and parent >= 0:
                        S1_credit[parent] += 1
                    
                    # Add to next time step
                    next_events.append((j, k, v + 1, i))
        
        if new_share_num == 0:
            break
        if t == T_max:
            print(f"Warning: Sharing activities did not stop naturally, reached T_max = {T_max}")
        
        current_events = next_events

    # Compute derived statistics
    S = X + Y
    S1 = S + S1_credit

    variables = {
        'V': V,
        'Z': Z,
        'X': X,
        'Y': Y,
        'R': R,
        'RT': RT,
        'RC': RC,
        'W': W,
        'S': S,
        'S1': S1
    }

    if verbose:
        total_bytes = sum(arr.nbytes for arr in [V, Z, X, Y, R, RT, RC, W, S, S1])
        print(f"\nMemory Usage: vectors={total_bytes/(1024*1024):.2f} MB")
    
    return variables


def simulate_social_sharing_hawkes(G, n, m, T_max, watching_params,
                            sharing_params, 
                            pi=1.0, p=0.5, seed=None,
                            randomization_unit='user', cluster_labels=None,
                            verbose=False):
    """
    Memory-optimized Hawkes process simulation for social sharing.
    
    Returns aggregated statistics instead of storing full watch history.
    
    Returns:
        dict with keys:
            V: experiment inclusion indicator
            Z: treatment assignment
            X: share-driven success count (user shared after receiving a share)
            Y: independent success count (user shared after independent watch)
            R: total share-driven watch count per user
            RT: share-driven watch from Treatment group senders
            RC: share-driven watch from Control group senders
            W: total watch count per user
            S: total share success count (S = X + Y)
            S1: share success with 1-step propagation credit
    """
    if seed is not None:
        np.random.seed(seed)

    g_bar = 2 * G.number_of_edges() / n

    V, Z = assign_treatment(
        n, pi, p, seed=seed,
        randomization_unit=randomization_unit,
        cluster_labels=cluster_labels
    )
    a_d = watching_params['a_d']
    b_d = watching_params['b_d']

    # Aggregated statistics (length-n vectors)
    X = np.zeros(n, dtype=np.int64)  # share success after share-driven watch
    Y = np.zeros(n, dtype=np.int64)  # share success after independent watch
    R = np.zeros(n, dtype=np.int64)  # total share-driven watch count
    RT = np.zeros(n, dtype=np.int64) # share-driven from Treatment
    RC = np.zeros(n, dtype=np.int64) # share-driven from Control
    RT_d = np.zeros(n, dtype=np.int64) # share-driven from Treatment via direct share
    RC_d = np.zeros(n, dtype=np.int64) # share-driven from Control via direct share
    W = np.zeros(n, dtype=np.int64)  # total watch count
    S1_credit = np.zeros(n, dtype=np.int64)  # 1-step propagation credit

    # Queue: (user, item, generation, parent_user or -1 if immigrant)
    # We only track immediate parent instead of full ancestor list to save memory
    Q = deque()

    # Safety guard
    max_events = int(n * 500)
    total_share_events = 0

    # Precompute neighbors
    neighbors = {i: list(G.neighbors(i)) for i in range(n)}

    # --- Step 1: immigrants (generation 0) ---
    if verbose:
        print(f"[hawkes/cluster] generating immigrants (vectorized)...")
    
    lam_matrix = np.outer(a_d, b_d) / m
    n_immigrants = np.random.poisson(lam_matrix)
    
    nonzero_i, nonzero_k = np.nonzero(n_immigrants)
    counts = n_immigrants[nonzero_i, nonzero_k]
    
    # Update W (total watch count)
    for idx in range(len(nonzero_i)):
        i, cnt = nonzero_i[idx], counts[idx]
        W[i] += cnt
    
    # Build queue
    total_immigrants = int(counts.sum())
    if total_immigrants > 0:
        expanded_i = np.repeat(nonzero_i, counts)
        expanded_k = np.repeat(nonzero_k, counts)
        
        for idx in range(min(total_immigrants, max_events)):
            # (user, item, generation, parent_user)
            Q.append((int(expanded_i[idx]), int(expanded_k[idx]), 0, -1))
                    
    # --- Step 2: branching (offspring) ---
    while Q and total_share_events < max_events:
        if verbose:
            print(f"Q size before pop: {len(Q)}")
        i, k, v, parent = Q.popleft()
        mode = 'd' if v == 0 else 's'
        group = get_group(V[i], Z[i])

        for j in neighbors.get(i, []):
            lam = get_share_prob(g_bar, i, j, k, sharing_params, group, mode=mode)
            n_off = np.random.poisson(lam)
            if n_off <= 0:
                continue

            for _ in range(n_off):
                total_share_events += 1
                
                # Update statistics for receiver j
                W[j] += 1
                R[j] += 1
                if V[i] == 1:
                    if Z[i] == 1:
                        RT[j] += 1
                    else:
                        RC[j] += 1
                    if v == 0:
                        if Z[i] == 1:
                            RT_d[j] += 1
                        else:
                            RC_d[j] += 1
                
                # Update statistics for sender i
                if v == 0:
                    Y[i] += 1  # i watched independently (gen 0) and caused a share
                else:
                    X[i] += 1  # i watched via share (gen > 0) and caused a share
                
                # Update 1-step propagation credit (for parent of sender i)
                if v >= 1 and parent >= 0:
                    S1_credit[parent] += 1
                
                # Add to queue
                Q.append((int(j), int(k), int(v) + 1, int(i)))
                
                if total_share_events >= max_events:
                    break
            if total_share_events >= max_events:
                break

    if verbose and total_share_events >= max_events:
        print(f"[hawkes/cluster] reached max_events={max_events}, cascades truncated")

    # Compute derived statistics
    S = X + Y  # total share success
    S1 = S + S1_credit  # share success with 1-step propagation

    variables = {
        'V': V,
        'Z': Z,
        'X': X,  # share_driven_success
        'Y': Y,  # indep_share_success  
        'R': R,  # share_driven_watch_count
        'RT': RT,
        'RC': RC,
        'RT_d': RT_d,
        'RC_d': RC_d,
        'W': W,  # total watch count
        'S': S,  # share_success = X + Y
        'S1': S1  # share_success_with_propagation
    }

    if verbose:
        total_bytes = sum(arr.nbytes for arr in [V, Z, X, Y, R, RT, RC, W, S, S1])
        print(f"\nMemory Usage: vectors={total_bytes/(1024*1024):.2f} MB")

    return variables


def simulate_social_sharing_hawkes_ct_exp_window(
    G,
    n,
    m,
    T_max,
    watching_params,
    sharing_params,
    obs_window_W=0.0,
    beta=1.0,
    pi=1.0,
    p=0.5,
    seed=None,
    simulate_beyond_window=False,
    randomization_unit='user',
    cluster_labels=None,
    verbose=False,
):
    """Continuous-time Hawkes branching simulation with exponential kernel.

    This implementation keeps the same aggregated outputs as existing simulators
    and adds event times with a finite observation window:
      T_eval = T_max + obs_window_W.

    Queue tuple format:
      (event_time, user, item, generation, parent_user)

        Switch mode:
            - simulate_beyond_window=False: generate and process only events needed for [0, T_eval]
            - simulate_beyond_window=True: continue propagation beyond T_eval,
                but record statistics only for events with event_time <= T_eval
    """
    if seed is not None:
        np.random.seed(seed)

    if not np.isfinite(T_max) or T_max <= 0:
        raise ValueError(f"T_max must be positive finite, got {T_max}.")
    if not np.isfinite(obs_window_W) or obs_window_W < 0:
        raise ValueError(f"obs_window_W must be non-negative finite, got {obs_window_W}.")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError(f"beta must be positive finite, got {beta}.")

    T_max = float(T_max)
    obs_window_W = float(obs_window_W)
    beta = float(beta)
    simulate_beyond_window = bool(simulate_beyond_window)
    T_eval = T_max + obs_window_W

    g_bar = 2 * G.number_of_edges() / n

    V, Z = assign_treatment(
        n,
        pi,
        p,
        seed=seed,
        randomization_unit=randomization_unit,
        cluster_labels=cluster_labels,
    )

    a_d = watching_params['a_d']
    b_d = watching_params['b_d']

    # Aggregated statistics (same semantics as existing simulators)
    X = np.zeros(n, dtype=np.int64)
    Y = np.zeros(n, dtype=np.int64)
    R = np.zeros(n, dtype=np.int64)
    RT = np.zeros(n, dtype=np.int64)
    RC = np.zeros(n, dtype=np.int64)
    RT_d = np.zeros(n, dtype=np.int64)
    RC_d = np.zeros(n, dtype=np.int64)
    watch_total = np.zeros(n, dtype=np.int64)
    S1_credit = np.zeros(n, dtype=np.int64)

    # Queue is a min-heap by event_time with tuples:
    # (event_time, user, item, generation, parent_user)
    Q = []

    # Safety guard for explosive cascades
    window_scale = max(T_eval / max(T_max, 1.0), 1.0)
    max_events = int(n * 500 * window_scale)
    total_share_events = 0
    recorded_share_events = 0

    neighbors = {i: list(G.neighbors(i)) for i in range(n)}

    # Step 1: independent (immigrant) events on [0, T_max]
    # Rate is aligned with current a_d/b_d structure: lambda_ik = a_d[i] * b_d[k] / m.
    immigrant_rate_matrix = np.outer(a_d, b_d) / (m * T_max)
    immigrant_mean_matrix = immigrant_rate_matrix * T_max
    n_immigrants = np.random.poisson(immigrant_mean_matrix)

    nonzero_i, nonzero_k = np.nonzero(n_immigrants)
    counts = n_immigrants[nonzero_i, nonzero_k]

    for idx in range(len(nonzero_i)):
        i, cnt = nonzero_i[idx], counts[idx]
        watch_total[i] += cnt

    total_immigrants = int(counts.sum())
    if total_immigrants > 0:
        expanded_i = np.repeat(nonzero_i, counts)
        expanded_k = np.repeat(nonzero_k, counts)
        immigrant_times = np.random.uniform(0.0, T_max, size=total_immigrants)

        Q = [
            (float(immigrant_times[idx]), int(expanded_i[idx]), int(expanded_k[idx]), 0, -1)
            for idx in range(total_immigrants)
        ]
        heapq.heapify(Q)

    # Step 2: branching events
    while Q and total_share_events < max_events:
        event_time, i, k, v, parent = heapq.heappop(Q)
        if (not simulate_beyond_window) and event_time > T_eval:
            break

        mode = 'd' if v == 0 else 's'
        group = get_group(V[i], Z[i])
        delta = T_eval - event_time
        if (not simulate_beyond_window) and delta <= 0:
            continue

        if not simulate_beyond_window:
            exp_neg_beta_delta = np.exp(-beta * delta)
            trunc_mass = 1.0 - exp_neg_beta_delta
            if trunc_mass <= 0:
                continue

        for j in neighbors.get(i, []):
            prob = get_share_prob(g_bar, i, j, k, sharing_params, group, mode=mode)
            if prob <= 0:
                continue

            # Using lambda(dt) = beta * prob * exp(-beta * dt), dt >= 0.
            # If simulate_beyond_window=False, sample from [0, delta].
            # If simulate_beyond_window=True, sample over [0, +inf).
            if simulate_beyond_window:
                expected_offspring = prob
            else:
                expected_offspring = prob * trunc_mass
            n_off = np.random.poisson(expected_offspring)
            if n_off <= 0:
                continue

            if simulate_beyond_window:
                dt = np.random.exponential(scale=1.0 / beta, size=n_off)
            else:
                # Sample offspring delays from truncated Exp(beta) on [0, delta].
                u = np.random.rand(n_off)
                dt = -np.log(1.0 - u * trunc_mass) / beta
            child_times = event_time + dt

            for child_time in child_times:
                total_share_events += 1

                is_observed = child_time <= T_eval
                if is_observed:
                    recorded_share_events += 1
                    watch_total[j] += 1
                    R[j] += 1
                    if V[i] == 1:
                        if Z[i] == 1:
                            RT[j] += 1
                        else:
                            RC[j] += 1
                        if v == 0:
                            if Z[i] == 1:
                                RT_d[j] += 1
                            else:
                                RC_d[j] += 1

                    if v == 0:
                        Y[i] += 1
                    else:
                        X[i] += 1

                    if v >= 1 and parent >= 0:
                        S1_credit[parent] += 1

                if simulate_beyond_window or is_observed:
                    heapq.heappush(Q, (float(child_time), int(j), int(k), int(v) + 1, int(i)))

                if total_share_events >= max_events:
                    break
            if total_share_events >= max_events:
                break

    if verbose:
        print(
            f"[hawkes_ct_exp_window] T_max={T_max:.4f}, W={obs_window_W:.4f}, "
            f"T_eval={T_eval:.4f}, immigrants={total_immigrants}, "
            f"share_events(simulated)={total_share_events}, "
            f"share_events(recorded)={recorded_share_events}, max_events={max_events}, "
            f"simulate_beyond_window={simulate_beyond_window}"
        )
        if total_share_events >= max_events:
            print("[hawkes_ct_exp_window] reached max_events, cascades truncated")

    S = X + Y
    S1 = S + S1_credit

    variables = {
        'V': V,
        'Z': Z,
        'X': X,
        'Y': Y,
        'R': R,
        'RT': RT,
        'RC': RC,
        'RT_d': RT_d,
        'RC_d': RC_d,
        'W': watch_total,
        'S': S,
        'S1': S1,
    }

    return variables



def diagnose_poisson_columns(matrix, alpha=0.05):
    """Diagnose whether each column follows a Poisson distribution.
    
    Args:
        matrix: (n_repeats x n) matrix, each column is a sample sequence
        alpha: significance level (default 0.05)
    """
    n_repeats, n = matrix.shape
    
    col_means = np.mean(matrix, axis=0)
    col_vars = np.var(matrix, axis=0, ddof=1)
    
    # For Poisson, mean should equal variance
    mean_var_ratio = col_vars / np.maximum(col_means, 1e-12)
    mean_var_diff = np.abs(col_vars - col_means)
    
    # Chi-squared goodness-of-fit test
    def poisson_goodness_of_fit(data):
        data = np.asarray(data)
        if data.size == 0:
            return np.nan, np.nan
        if np.any(data < 0):
            raise ValueError("data contains negative values.")
        if len(np.unique(data)) < 2:
            return np.nan, np.nan

        lambda_hat = data.mean()

        k_max = int(max(data.max(), np.ceil(lambda_hat + 5 * np.sqrt(lambda_hat))))
        k_vals = np.arange(k_max + 1)

        # Observed frequencies
        observed = np.bincount(data.astype(int), minlength=k_max + 1)

        # Expected frequencies
        pmf = stats.poisson.pmf(k_vals, lambda_hat)
        expected = data.size * pmf

        obs = observed.tolist()
        exp = expected.tolist()

        def merge_three_lists_by_threshold(obs, exp, threshold=5):
            """Merge frequency bins so that each bin has expected count >= threshold.
            
            Args:
                obs: observed frequency list
                exp: expected frequency list
                threshold: minimum expected count per bin (default 5)
            
            Returns:
                (merged_obs, merged_exp): merged frequency lists
            """
            if not exp or len(exp) == 0:
                return obs.copy(), exp.copy()
            
            if threshold <= 0:
                return obs.copy(), exp.copy()
            
            merged_obs = []
            merged_exp = []
            current_obs_sum = 0
            current_exp_sum = 0
            
            for i in range(len(exp)):
                current_obs_sum += obs[i]
                current_exp_sum += exp[i]
                
                if current_exp_sum >= threshold:
                    merged_obs.append(current_obs_sum)
                    merged_exp.append(current_exp_sum)
                    current_obs_sum = 0
                    current_exp_sum = 0
            
            # Merge remainder into last bin
            if current_exp_sum > 0:
                if merged_exp:
                    merged_obs[-1] += current_obs_sum
                    merged_exp[-1] += current_exp_sum
                else:
                    merged_obs.append(current_obs_sum)
                    merged_exp.append(current_exp_sum)
            
            return merged_obs, merged_exp
        
        obs, exp = merge_three_lists_by_threshold(obs, exp, threshold=5)

        k_bins = len(exp)
        if k_bins <= 2:
            return np.nan, np.nan
        df = k_bins - 1 - 1

        chi2_stat = np.sum((np.array(obs) - np.array(exp))**2 / np.array(exp))

        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        return chi2_stat, p_value
    
    # Test each column
    chi2_stats = []
    p_values = []
    for j in range(n):
        chi2, p_val = poisson_goodness_of_fit(matrix[:, j])
        chi2_stats.append(chi2)
        p_values.append(p_val)
    
    chi2_stats = np.array(chi2_stats)
    p_values = np.array(p_values)
    
    # Summary statistics
    valid_tests = ~np.isnan(p_values)
    poisson_like = (p_values > alpha) & valid_tests
    good_mean_var = np.abs(mean_var_ratio - 1) < 0.1  # mean-variance ratio close to 1
    
    return pd.DataFrame({
        'col_means': col_means,
        'col_vars': col_vars,
        'mean_var_diff': mean_var_diff,
        'mean_var_ratio': mean_var_ratio,
        'good_mean_var': good_mean_var,
        'valid_tests': valid_tests,
        'p_values': p_values,
        'poisson_like': poisson_like
    })


if __name__ == "__main__":
    from configs.config import (
        T_MAX, N_REPEATS, MAX_WORKERS, NETWORK_PARAMS,
        POPULATION_PARAMS, PARAMS_RELATED, RANDOM_SEEDS,
        HOMO_VALUES, HETERO_RANGES
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
    
    ##### Generate network #####
    G = generate_network(
        n,
        seed=RANDOM_SEEDS['seed_graph'],
        **NETWORK_PARAMS
    )
    n = len(G.nodes())
    
    ##### Sample parameters #####
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

    start_time = time.time()
    variables = simulate_social_sharing_hawkes(G, n, m,T_max, watching_params,
                                                         sharing_params, 
                                                         pi, p, seed = RANDOM_SEEDS['seed_base'], 
                                                         verbose=True)
    end_time = time.time()
    print(f"\nTime cost = {end_time - start_time:.3f} secs")


