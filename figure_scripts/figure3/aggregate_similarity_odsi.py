from __future__ import annotations

import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from figure_scripts.figure3.generate_similarity_panels_odsi import main


if __name__ == "__main__":
    main()
