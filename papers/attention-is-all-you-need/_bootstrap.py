"""Put the repo root on sys.path so `import utils` works when a paper script is run
from its own folder (`python train.py`). Import this before any `utils` import."""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
