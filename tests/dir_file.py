import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.dir_file import DirFileManager


def test_ensure_directory_creates_nested_results_dir(tmp_path):
    # Mirrors eval_encoder.py: metric_results.json is written into a nested
    # results_dir that does not exist yet on a fresh run. open(..., "w") does
    # not create parent dirs, so the save must be preceded by this call.
    results_dir = tmp_path / "run" / "ptbxl_0.5_512_None_4"
    assert not results_dir.exists()

    DirFileManager.ensure_directory_exists(folder=results_dir)
    assert results_dir.is_dir()

    out = results_dir / "metric_results.json"
    with open(out, "w") as f:
        json.dump({"mse": 0.1}, f)
    assert out.exists()
