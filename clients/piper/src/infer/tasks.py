"""Select model -> task -> checkpoint before GPU or hardware initialization."""
import argparse
import hashlib
import json
from pathlib import Path

from .model.api import CONTRACT
from .model.loader import read_config

MODEL_DIR = Path(__file__).resolve().parent / 'model'
REGISTRY = MODEL_DIR / 'models.json'


def model_registry():
    registry = json.loads(REGISTRY.read_text())
    if not isinstance(registry, dict) or not registry:
        raise ValueError('models.json must contain model definitions')
    return registry


def require_model(model):
    if not model:
        raise ValueError('--model is required')
    registry = model_registry()
    if model not in registry:
        raise ValueError(f'Unknown model {model!r}; choose from: {", ".join(registry)}')
    entry = registry[model]
    if entry.get('implemented') is not True:
        if model == 'fastwam':
            from .model.fastwam import NOT_IMPLEMENTED
            raise NotImplementedError(NOT_IMPLEMENTED)
        raise NotImplementedError(f'{model}: backend not implemented')
    return entry


def task_registry(model):
    return require_model(model)['tasks']


def get_task(name, model):
    tasks = task_registry(model)
    if name not in tasks:
        raise ValueError(f'Unknown task {name!r} for {model}; choose from: {", ".join(tasks)}')
    path = (MODEL_DIR / tasks[name]).resolve()
    result = json.loads(path.read_text())
    result['task_dir'] = path.parent
    default = result.get('default_checkpoint')
    if not default or default not in result.get('checkpoints', {}):
        raise ValueError(f'{model}/{name}: default checkpoint is not configured')
    result['config'] = (path.parent / result['checkpoints'][default]).resolve()
    return result


def checkpoint_config(task, checkpoint, model):
    entry = get_task(task, model)
    if checkpoint in entry['checkpoints']:
        return (entry['task_dir'] / entry['checkpoints'][checkpoint]).resolve()
    path = Path(checkpoint).expanduser().resolve()
    if path.is_file() and path.suffix == '.json':
        return path
    if path.is_dir() and (path / 'deployment.json').is_file():
        manifest = path / 'deployment.json'
        if read_config(manifest)['checkpoint_dir'] != path:
            raise ValueError('Bundle deployment.json must reference its own checkpoint directory')
        return manifest
    raise ValueError(f'Checkpoint {checkpoint!r} is not registered for {model}/{task}; '
                     'provide a deployment JSON or a bundle containing deployment.json')


def select_server_task(task=None, config=None, checkpoint=None, *, model=None):
    require_model(model)
    if not task:
        raise ValueError('--task is required')
    entry = get_task(task, model)
    if config and checkpoint:
        raise ValueError('Use either --config or --checkpoint, not both')
    if config:
        path = Path(config).expanduser().resolve()
    else:
        path = checkpoint_config(task, checkpoint, model) if checkpoint else entry['config']
    deployment_metadata(path, task, model)
    return task, path


def deployment_metadata(config_path, task=None, model=None):
    config = read_config(config_path)
    required = ('model', 'backend', 'task', 'checkpoint_id', 'camera_mode', 'state_dim',
                'action_dim', 'action_horizon', 'contract', 'checkpoint_dir', 'train_config')
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f'Deployment metadata missing: {", ".join(missing)}')
    if model and config['model'] != model:
        raise ValueError(f"Model/config mismatch: requested {model}, configured {config['model']}")
    backend = require_model(config['model'])['backend']
    if config['backend'] != backend:
        raise ValueError(f"Model/backend mismatch: {config['model']} requires {backend}")
    if task and config['task'] != task:
        raise ValueError(f"Task/config mismatch: requested {task}, configured {config['task']}")
    if (config['contract'] != CONTRACT or config['state_dim'] != 23 or config['action_dim'] != 23
            or config['camera_mode'] not in ('active_only', 'three_cameras')
            or type(config['action_horizon']) is not int or not 1 <= config['action_horizon'] <= 256
            or config.get('action_dt_s') != .05):
        raise ValueError('Deployment must use Piper23 absolute EEF at 20Hz and a supported camera mode')
    serialized = json.dumps(config, sort_keys=True, default=str, separators=(',', ':'))
    metadata = {key: config[key] for key in required}
    metadata['checkpoint_dir'] = str(metadata['checkpoint_dir'])
    metadata['dt_s'] = config['action_dt_s']
    metadata['deployment_id'] = hashlib.sha256(serialized.encode()).hexdigest()
    return metadata


def check_task_health(health, expected, checkpoint=None, *, model):
    require_model(model)
    if health.get('model') != model:
        raise ValueError(f"Server model mismatch: expected {model}, got {health.get('model')!r}")
    if health.get('task') != expected:
        raise ValueError(f"Server task mismatch: expected {expected}, got {health.get('task')!r}")
    if checkpoint:
        task, config = select_server_task(expected, checkpoint=checkpoint, model=model)
        expected_id = deployment_metadata(config, task, model)['deployment_id']
        if (health.get('deployment') or {}).get('deployment_id') != expected_id:
            raise ValueError('Server checkpoint mismatch; restart Server with the requested --checkpoint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--task')
    parser.add_argument('--checkpoint')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--field', choices=('config', 'prompt'))
    args = parser.parse_args()
    if args.list:
        registry = model_registry()
        print(json.dumps({args.model:registry[args.model]} if args.model else registry,
                         ensure_ascii=False, indent=2))
        return
    if not args.model or not args.task:
        parser.error('--model and --task are required')
    if args.field == 'prompt':
        print(get_task(args.task, args.model)['prompt'])
    else:
        task, config = select_server_task(args.task, checkpoint=args.checkpoint, model=args.model)
        print(config if args.field == 'config' else json.dumps(deployment_metadata(config, task, args.model), indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, NotImplementedError) as exc:
        raise SystemExit(str(exc))
