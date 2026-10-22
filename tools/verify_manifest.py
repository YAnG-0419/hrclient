"""Verify archive contents against MANIFEST.json without executing client code."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "MANIFEST.json").read_text())
expected = set()
for item in manifest["files"]:
    relative = Path(item["path"])
    if relative.is_absolute() or ".." in relative.parts or item["path"] in expected:
        raise SystemExit(f"Unsafe or duplicate path: {relative}")
    expected.add(item["path"])
    file = root / relative
    if file.is_symlink() or not file.is_file():
        raise SystemExit(f"Missing or symlink: {relative}")
    content = file.read_bytes()
    if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
        raise SystemExit(f"Hash/size mismatch: {relative}")
actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
          and "__pycache__" not in p.parts and p.suffix != ".pyc"}
if actual != expected | {"MANIFEST.json"}:
    raise SystemExit(f"Unlisted files: {sorted(actual - expected - {'MANIFEST.json'})}")
print(f"VERIFIED: {len(expected)} files; all sizes and SHA256 match")
