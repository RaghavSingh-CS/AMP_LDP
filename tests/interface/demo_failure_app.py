"""Browser-only error fixture. Production app/data are never modified.

Run explicitly with Streamlit on a separate local port, then inspect a demo
preset. This simulates only missing history access while keeping genuine scores.
"""

from pathlib import Path
import runpy
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data_loader import ArtifactError

with patch("app.views.inspector.load_history", side_effect=ArtifactError(
        "TEST FIXTURE: original history source is unavailable: data/raw/data set.csv")):
    runpy.run_path(str(ROOT / "app/main.py"), run_name="__main__")
