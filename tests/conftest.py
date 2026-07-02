import pathlib
import sys

MONOGS_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(MONOGS_ROOT) not in sys.path:
    sys.path.insert(0, str(MONOGS_ROOT))
