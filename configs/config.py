"""Configuration parameters for the FlyingWheel simulation"""

T_MAX = 24
N_REPEATS = 100
MAX_WORKERS = 10
SIMULATION_METHOD = 'hawkes'  # 'discrete', 'hawkes' or 'hawkes_ct_exp_window'

# Optional parameters for simulation_method='hawkes_ct_exp_window'
# T_eval = T_MAX + W, exponential kernel decay rate is beta.
HAWKES_CT_WINDOW = {
    'W': 1000.0,
    'beta': 1.0,
    'simulate_beyond_window': True,
}

# Estimator type: 'ht' (Horvitz-Thompson) or 'hajek'
# HT: uses n*p*pi as denominator (theoretical expected sample size)
# Hajek: uses actual sample counts as denominator
ESTIMATOR_TYPE = 'hajek'

# Multiprocessing start method
# 'fork': Linux only, child processes share memory (more efficient for large data)
# 'spawn': Windows/macOS/Linux, child processes get fresh copies (safer but more memory)
# None: use system default
MP_START_METHOD = None  # Set to 'spawn' for Windows testing

# Whether to use homogeneous parameters (same for all users) or heterogeneous (sampled from distribution).
# homogeneous parameters if True
PARAMS_RELATED = {
    'is_homo': False,
}

# Cluster randomization settings for the cluster baseline estimator
CLUSTER_RANDOMIZATION = {
    'method': 'leiden',     # 'louvain' or 'leiden'
    'resolution': 1.0, # resolution is larger, more communities
    'seed': 123
}

# Network parameters (used in scripts that read NETWORK_PARAMS)
NETWORK_PARAMS = {
    # Supported methods:
    # 'barabasi', 'watts', 'regular', 'sbm', 'er', 'fbSND', 'brightkite', 'twitch_gamer', 'abc'
    'method': 'barabasi',

    # Shared for degree-based synthetic graphs ('barabasi', 'watts', 'regular').
    # For barabasi this is the target average degree, internally converted to BA's m parameter.
    'm_edges': 50,

    # Watts-Strogatz parameter (only used when method='watts').
    'graph_p': 0.2,

    # Erdos-Renyi parameter (only used when method='er').
    # If omitted in calls, generate_network will fall back to graph_p.
    'er_p': 0.02,

    # Stochastic Block Model parameters (only used when method='sbm').
    'sbm_k': 3,
    # Homogeneous SBM probabilities are derived from:
    #   p_out / p_in = sbm_eta
    #   sbm_d = (base_size - 1) * p_in + (n - base_size) * p_out
    # where base_size = n // sbm_k.
    'sbm_d': 10.0,
    'sbm_eta': 0.2,
    # Optional override: provide a full KxK probability matrix with entries in [0, 1].
    # If provided, sbm_d/sbm_eta are ignored.
    # 'sbm_probs': [[0.2, 0.05, 0.05], [0.05, 0.2, 0.05], [0.05, 0.05, 0.2]],
}

# Population parameters
POPULATION_PARAMS = {
    'n': 100,         # number of users
    'm': 200,         # number of videos
    'pi': 0.5,         # eligible probability
    'p': 0.5,      # treatment probability given eligible
}

# Random seeds
RANDOM_SEEDS = {
    'seed_params': 42,
    'seed_graph': 43,
    'seed_base': 46,
}

# Parameter values for homogeneous case
HOMO_VALUES = {
    'watching': {
        'a_d': 5,
        'b_d': 10
    },
    'sharing_T': {
        'phi': 0.5,       # base value for phi_d and phi_s
        'varphi': 0.1,    # base value for varphi_d and varphi_s
        'theta': 1.0,     # base value for theta_d and theta_s
    },
    # perturbation from _d to _s: _s = _d + perturbation
    'sharing_ds_perturbation': {
        'phi': 0.0,       # phi_s = phi_d + perturbation
        'varphi': 0.0,    # varphi_s = varphi_d + perturbation
        'theta': 0.0,     # theta_s = theta_d + perturbation
    },
    # shift for C and O groups (applied to both _d and _s)
    'sharing_offsets': {
        'phi': {'C': -0.1, 'O': -0.1},
        'varphi': {'C': 0, 'O': 0},
        'theta': {'C': 0.0, 'O': 0.0},
    }
}

# Parameter ranges for heterogeneous case
HETERO_RANGES = {
    'watching': {
        'a_d': (0, 10),
        'b_d': (0, 20)
    },
    'sharing_T': {
        'phi': (0, 1.0),      # range for phi_d (base)
        'varphi': (0, 0.2),   # range for varphi_d (base)
        'theta': (0, 1.0),    # range for theta_d (base)
    },
    # perturbation range from _d to _s: _s = _d + U(perturbation)
    'sharing_ds_perturbation': {
        'phi': (-0.1, 0.3),     # phi_s = phi_d + U(-0.1, 0.3)
        'varphi': (0.0, 0.0),   # varphi_s = varphi_d + U(-0.05, 0.1)
        'theta': (0.0, 0.0),   # theta_s = theta_d + U(-0.2, 0.4)
    },
    # shift ranges for C and O groups (applied to both _d and _s)
    'sharing_shift_ranges': {
        'phi': {'C': (-0.5, 0.0), 'O': (-0.5, 0.0)},
        'varphi': {'C': (0.0, 0.0), 'O': (0.0, 0.0)},
        'theta': {'C': (0.0, 0.0), 'O': (0.0, 0.0)},
    }
}