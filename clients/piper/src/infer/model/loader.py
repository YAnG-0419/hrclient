"""Common configuration validation and explicit backend dispatch; no GPU on import."""
import json
import math
from pathlib import Path


def read_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('configuration must be an object')
    for key in ('code_dir', 'checkpoint_dir', 'io_adapter', 'openpi_data_home'):
        value = config.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'invalid {key}')
            config[key] = (path.parent / value).resolve()
    return config


def configuration_issues(config):
    issues = []
    if config.get('model') == 'fastwam':
        from .fastwam import NOT_IMPLEMENTED
        return [NOT_IMPLEMENTED]
    if config.get('model') != 'pi05':
        issues.append('model must explicitly declare pi05; FastWAM is not implemented')
    if config.get('backend') != 'pi05_openpi':
        issues.append('backend must be pi05_openpi for model pi05')
    code = config.get('code_dir')
    if not code or not (code / 'src/openpi/policies/policy_config.py').is_file():
        issues.append('code_dir: missing OpenPI source')
    checkpoint = config.get('checkpoint_dir')
    if not checkpoint or not checkpoint.is_dir():
        issues.append('checkpoint_dir: provide the complete local checkpoint directory')
    else:
        if config.get('require_verified_checkpoint') and not (checkpoint / 'VERIFIED').is_file():
            issues.append('checkpoint_dir: SHA256 verification is pending (VERIFIED missing)')
        if not (checkpoint / 'params').is_dir() and not (checkpoint / 'model.safetensors').is_file():
            issues.append('checkpoint_dir: missing params/ or model.safetensors')
        if not (checkpoint / 'assets').is_dir():
            issues.append('checkpoint assets/ normalization resources missing')
    if not isinstance(config.get('train_config'), str) or not config['train_config'].strip():
        issues.append('train_config: provide the matching training config name')
    adapter = config.get('io_adapter')
    if not adapter or not adapter.is_file():
        issues.append('io_adapter: input/output definitions not configured')
    dt = config.get('action_dt_s')
    if isinstance(dt, bool) or not isinstance(dt, (int,float)) or not math.isfinite(dt) or dt <= 0:
        issues.append('action_dt_s: provide the training action interval in seconds')
    if not isinstance(config.get('sample_kwargs', {}), dict):
        issues.append('sample_kwargs must be an object')
    if not isinstance(config.get('device'), str) or not config['device'].strip():
        issues.append('device missing')
    return issues


def load_model(path):
    config = read_config(path)
    if config.get('model') == 'fastwam':
        from .fastwam import load_policy
        return load_policy(config)
    issues = configuration_issues(config)
    if issues:
        raise ValueError('NOT_READY:\n- ' + '\n- '.join(issues))
    if config.get('task'):
        from ..tasks import deployment_metadata
        deployment_metadata(path, config['task'], config['model'])
    from .pi05.loader import load_policy
    return load_policy(config)
