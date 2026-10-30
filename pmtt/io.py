"""Portable output serialization and deterministic identity keys."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import torch

def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def seed(*parts):
    return int(identity(parts)[:8], 16) % 2147483647

def destination(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def save(value, path):
    path = destination(path)
    temporary = path.with_name(path.name + '.partial')
    if path.suffix == '.json':
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    else:
        torch.save(value, temporary)
    os.replace(temporary, path)

def load(path):
    path = Path(path)
    if path.suffix == '.json':
        return json.loads(path.read_text(encoding='utf8'))
    return torch.load(path, map_location='cpu', weights_only=False)

def pack(value, path):
    path = destination(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as stream:
            torch.save(value, stream)
    os.replace(temporary, path)

def unpack(path):
    with gzip.open(path, 'rb') as stream:
        return torch.load(io.BytesIO(stream.read()), map_location='cpu', weights_only=False)

def ensure_finite(value):
    if not torch.isfinite(value).all():
        raise FloatingPointError('Non-finite tensor')
    return value
