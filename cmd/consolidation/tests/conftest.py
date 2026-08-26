"""pytest bootstrap: make the `consolidation` package importable regardless
of pytest's rootdir/invocation directory."""
import os
import sys

# cmd/ dir — adding it makes `import consolidation` (cmd/consolidation) work.
_CMD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _CMD_DIR not in sys.path:
    sys.path.insert(0, _CMD_DIR)
