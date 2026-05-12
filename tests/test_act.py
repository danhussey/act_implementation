import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def write_tiny_robomimic(path: Path) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i in range(3):
            demo = data.create_group(f"demo_{i}")
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_eef_pos", data=rng.normal(size=(16, 3)).astype(np.float32))
            obs.create_dataset("robot0_eef_quat", data=rng.normal(size=(16, 4)).astype(np.float32))
            obs.create_dataset("robot0_gripper_qpos", data=rng.normal(size=(16, 2)).astype(np.float32))
            demo.create_dataset("actions", data=rng.normal(size=(16, 7)).astype(np.float32))


def test_one_file_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "robomimic.hdf5"
    out = tmp_path / "run"
    write_tiny_robomimic(data)
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
            "--max-demos",
            "3",
        ],
        cwd=repo,
        check=True,
    )

    assert (out / "best.pt").exists()
    assert (out / "metrics.json").exists()


def test_walkthrough_notebook_is_valid_json() -> None:
    repo = Path(__file__).resolve().parents[1]
    notebook = json.loads((repo / "ACT_walkthrough.ipynb").read_text())

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) > 5
    assert any("state_dim" in "".join(cell["source"]) for cell in notebook["cells"])
