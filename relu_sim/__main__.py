"""Allow `python -m relu_sim`."""

import sys

from .app import run

sys.exit(run(sys.argv))
