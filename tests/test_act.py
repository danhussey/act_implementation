import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import act


def write_tiny_robomimic(path: Path, include_images: bool = False) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i in range(3):
            demo = data.create_group(f"demo_{i}")
            obs = demo.create_group("obs")
            obs.create_dataset("robot0_eef_pos", data=rng.normal(size=(16, 3)).astype(np.float32))
            obs.create_dataset("robot0_eef_quat", data=rng.normal(size=(16, 4)).astype(np.float32))
            obs.create_dataset("robot0_gripper_qpos", data=rng.normal(size=(16, 2)).astype(np.float32))
            obs.create_dataset("object", data=rng.normal(size=(16, 10)).astype(np.float32))
            if include_images:
                obs.create_dataset("agentview_image", data=rng.integers(0, 255, size=(16, 16, 16, 3), dtype=np.uint8))
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


def test_image_mode_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "robomimic_image.hdf5"
    out = tmp_path / "vision_run"
    write_tiny_robomimic(data, include_images=True)
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
            "--obs-mode",
            "image",
            "--vision-backbone",
            "scratch_cnn",
        ],
        cwd=repo,
        check=True,
    )

    assert (out / "best.pt").exists()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["obs_mode"] == "image"
    assert metrics["vision_backbone"] == "scratch_cnn"
    assets = act.load_rollout_assets(out / "best.pt", data, torch.device("cpu"))
    assert assets.obs_mode == "image"
    assert assets.image_shape == (3, 16, 16)


def test_image_mode_excludes_privileged_object_state(tmp_path: Path) -> None:
    data = tmp_path / "robomimic_image.hdf5"
    write_tiny_robomimic(data, include_images=True)

    dataset = act.DemoDataset(data, chunk_size=4, max_demos=3, obs_mode="image")

    assert "object" not in dataset.obs_keys
    assert dataset.obs_keys == ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    assert dataset.state_dim == 9
    assert dataset.image_shape == (3, 16, 16)


def test_plot_history_writes_rollout_curve_when_present(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "history.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"epoch": 1, "train_loss": 0.9, "val_loss": 1.1, "elapsed_seconds": 10.0}),
                json.dumps({"epoch": 2, "train_loss": 0.7, "val_loss": 0.8, "elapsed_seconds": 20.0}),
            ]
        )
    )
    (run / "rollout_history.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"epoch": 0, "successes": 1, "episodes": 4, "success_rate": 0.25}),
                json.dumps({"epoch": 2, "successes": 3, "episodes": 4, "success_rate": 0.75}),
            ]
        )
    )

    act.plot_history(SimpleNamespace(run=str(run), history=None, out=None, summary=None, title="tiny"))

    rollout_summary = json.loads((run / "rollout_summary.json").read_text())
    assert (run / "loss_curve.svg").exists()
    assert (run / "rollout_curve.svg").exists()
    assert rollout_summary["best_rollout_success_rate"] == 0.75


def test_walkthrough_notebook_is_valid_json() -> None:
    repo = Path(__file__).resolve().parents[1]
    notebook = json.loads((repo / "ACT_walkthrough.ipynb").read_text())

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) > 5
    assert any("state_dim" in "".join(cell["source"]) for cell in notebook["cells"])
