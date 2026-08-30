import yaml


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _parse_value(value):
    lowered = value.strip().lower()
    if lowered in ('true', 'false'):
        return lowered == 'true'
    if lowered in ('none', 'null'):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _set_nested(cfg, keys, value):
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def apply_overrides(cfg, overrides):
    for item in overrides or []:
        key, _, value = item.partition('=')
        if not key or not value:
            raise ValueError('Invalid override %r, expected key=value' % (item,))
        _set_nested(cfg, key.strip().split('.'), _parse_value(value))
