import subprocess
import sys
from pathlib import Path


def test_one_file_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "mock.hdf5"
    out = tmp_path / "run"

    subprocess.run(
        [sys.executable, "act.py", "mock", "--out", str(data), "--episodes", "3", "--steps", "16", "--image-size", "32"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "act.py",
            "train",
            "--data",
            str(data),
            "--out",
            str(out),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--chunk-size",
            "4",
            "--dim",
            "32",
        ],
        cwd=repo,
        check=True,
    )

    assert (out / "best.pt").exists()
    assert (out / "metrics.json").exists()
