"""Check a packaged release's source files before executing its entry point."""
import hashlib
import json
from pathlib import Path


def verify_release(root):
    root=Path(root).resolve()
    manifest_path=root/'release_manifest.json'
    if not manifest_path.exists():
        raise ValueError('A versioned navigation release manifest is required on the robot')
    manifest=json.loads(manifest_path.read_text())
    for name,digest in manifest['files'].items():
        path=(root/name).resolve()
        if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Navigation release source differs: '+name)
    return manifest
