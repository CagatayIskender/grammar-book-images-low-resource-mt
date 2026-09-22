"""Resolve recorded metric paths without rewriting frozen experiment identities."""
import hashlib
import json
from pathlib import Path


def metric_path(root, recorded):
    root = Path(root).resolve()
    path = Path(recorded)
    if path.is_absolute():
        path = path.relative_to(root)
    if '..' in path.parts:
        raise ValueError('Parent traversal is not an artifact path')
    old = Path('metrics/matched_gold_v2')
    xxl = Path('metrics/matched_gold_v2_xcomet_xxl')
    if path.is_relative_to(xxl):
        path = old / 'xcomet_xxl' / path.relative_to(xxl)
    elif path.is_relative_to(old):
        suffix = path.relative_to(old)
        if suffix.parts and suffix.parts[0] not in ('lexical_and_xcomet_xl', 'xcomet_xxl'):
            path = old / 'lexical_and_xcomet_xl' / suffix
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('Artifact escaped the repository')
    return resolved


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _migration(root, name):
    manifest = Path(root) / 'docs/metric_layout_migration.json'
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text())
    for path, expected in data.get('runtime_support', {}).items():
        if _sha(Path(root) / path) != expected:
            raise ValueError(f'Migration helper hash mismatch: {path}')
    return next((r for r in data['code_migrations'] if r['path'] == name), None)


def recorded_code_path(root, name, expected):
    """Validate both archived production code and the approved path-only adapter."""
    root = Path(root).resolve()
    current = root / name
    if _sha(current) == expected:
        return current
    entry = _migration(root, name)
    if not entry or entry['original_sha256'] != expected:
        raise ValueError(f'Unrecognized frozen-code change: {name}')
    snapshot = (root / entry['snapshot']).resolve()
    if not snapshot.is_relative_to(root / 'archive/code_before_metric_layout'):
        raise ValueError('Code snapshot escaped the archive')
    if _sha(snapshot) != expected or _sha(current) != entry['runtime_sha256']:
        raise ValueError(f'Migration code hash mismatch: {name}')
    return snapshot


def equivalent_scorer(root, name, saved, current):
    if saved == current:
        return True
    entry = _migration(root, name)
    if not entry or saved != entry['original_sha256'] or current != entry['runtime_sha256']:
        return False
    try:
        recorded_code_path(root, name, saved)
    except (OSError, ValueError, KeyError):
        return False
    return True
