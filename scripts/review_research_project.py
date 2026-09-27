"""Review a saved Brilliance project without starting the app or calling a model."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brilliance.research_review import main

if __name__ == "__main__":
    raise SystemExit(main())
