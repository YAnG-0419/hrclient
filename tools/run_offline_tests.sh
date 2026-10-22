#!/usr/bin/env bash
# Loopback sockets only; no SSH, model server, GPU or robot IO.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python3}
python_bin=$(command -v "$python_bin")
python_bin=$("$python_bin" -c 'import sys; print(sys.executable)')
(
  cd "$root/clients/g2"
  PYTHONPATH="$PWD/scripts" "$python_bin" -m unittest discover -s tests -p test_mixed76_v4.py -v
)
(
  cd "$root/clients/piper"
  PYTHONPATH="$PWD:$PWD/src/infer/mixed76_common" "$python_bin" -m unittest discover -s src/infer/mixed76_common -p test_unified_client.py -v
)
(
  cd "$root/clients/fr3_wuji"
  PYTHONPATH="$PWD/scripts" "$python_bin" -m unittest discover -s tests -p 'test_*.py' -v
)
