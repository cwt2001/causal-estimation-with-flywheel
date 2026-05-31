import importlib


def load_config_module(config_ref):
    """Load a Python config module from module path.

    Parameters
    ----------
    config_ref : str
        Module path like 'configs.config'.

    Returns
    -------
    tuple
        (module, source_info) where source_info describes the source reference.
    """
    if not config_ref or not str(config_ref).strip():
        raise ValueError("config reference cannot be empty")

    config_ref = str(config_ref).strip()

    module = importlib.import_module(config_ref)
    source_info = {
        "input": config_ref,
        "type": "module",
    }
    return module, source_info


def require_config_attrs(config_module, required_names, source_label="config"):
    """Extract required uppercase config attributes from a module."""
    missing = [name for name in required_names if not hasattr(config_module, name)]
    if missing:
        missing_str = ", ".join(missing)
        raise AttributeError(f"Missing required config fields in {source_label}: {missing_str}")

    return {name: getattr(config_module, name) for name in required_names}
