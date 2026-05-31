EXPERIMENT_SETTING = {
    # Name used in saved metadata.
    'name': 'watts_uniform_sweep_rewiring_p',

    # Plot x-axis config.
    'x_sweep_key': 'graph',
    'x_axis_label': 'rewiring p',
    'x_value_order': [
        'watts_d50_p05_n50000',
        'watts_d50_p10_n50000',
        'watts_d50_p30_n50000',
    ],
    'x_value_labels': {
        'watts_d50_p05_n50000': '0.05',
        'watts_d50_p10_n50000': '0.10',
        'watts_d50_p30_n50000': '0.30',
    },

    # Base simulation config module.
    'base_config_path': 'configs.exp_config',

    # Sweep video count.
    'm_configs': [
        {'m': 2000, 'name': 'm2000'},
    ],

    # Sweep: graph type, sample size n, and graph-specific parameters.
    # For regular/watts/barabasi, use m_edges as target average degree.
    # For sbm, use sbm_k/sbm_d/sbm_eta (do not use m_edges).
    # Use a unified key set below; irrelevant keys can stay None.
    'graph_configs': [
        {
            'n': 50000,
            'method': 'watts',
            'm_edges': 50,
            'graph_p': 0.05,
            'er_p': None,
            'sbm_k': None,
            'sbm_d': None,
            'sbm_eta': None,
            'sbm_probs': None,
            'name': 'watts_d50_p05_n50000'
        },
        {
            'n': 50000,
            'method': 'watts',
            'm_edges': 50,
            'graph_p': 0.1,
            'er_p': None,
            'sbm_k': None,
            'sbm_d': None,
            'sbm_eta': None,
            'sbm_probs': None,
            'name': 'watts_d50_p10_n50000'
        },
        {
            'n': 50000,
            'method': 'watts',
            'm_edges': 50,
            'graph_p': 0.3,
            'er_p': None,
            'sbm_k': None,
            'sbm_d': None,
            'sbm_eta': None,
            'sbm_probs': None,
            'name': 'watts_d50_p30_n50000'
        }
    ],


    # Sweep: disturbance level via shift ranges.
    'phi_varphi_theta_shifts': [
        {'phi': (0.0, 0.0), 'varphi': (0.0, 0.0), 'theta': (0.0, 0.0), 'name': 'phi_0_0_var_0_0_t_0_0'},
        {'phi': (-0.5, 0.0), 'varphi': (0.0, 0.0), 'theta': (0.0, 0.0), 'name': 'phi_m05_0_var_0_0_t_0_0'},
    ],

    # Sweep: exposure probability.
    'pi_values': [0.5],

    # Sweep: baseline parameter distribution in hetero mode.
    # This only controls a_d, b_d, phi_d(T), varphi_d(T), theta_d(T).
    # ds_perturbation and shift_ranges stay uniform as defined in base config.
    'random_para_dist_configs': [
        {
            'name': 'uniform_baseline',
            'dist': 'uniform',
            'c_by_param': {
                'a_d': 10.0,
                'b_d': 20.0,
                'phi': 1.0,
                'varphi': 0.2,
                'theta': 1.0,
            }
        },
    ],
}
