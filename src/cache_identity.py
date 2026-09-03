"""Content-based identities for data-dependent runtime caches."""
import hashlib
import json


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported cache identity value: {type(value).__name__}")


def content_digest(payload):
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), default=_json_value)
    for chunk in encoder.iterencode(payload):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
