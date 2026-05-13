import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH for local imports in tests
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
