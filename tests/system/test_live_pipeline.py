"""model prediction & output system test.

Launches the real `video_publisher` ROS2 node (replaying a tiny synthetic
clip) together with `slam.py --config configs/live/ROS.yaml`, and asserts:
  - both processes start and run without crashing
  - the model produces output: /monoGS/trajectory receives growing pose data
    and /monoGS/cloud receives at least one non-empty point cloud
  - both processes can be told to stop and exit within a grace period (no
    hang / zombie process) when sent SIGINT

Requires a CUDA GPU (torch + the built diff-gaussian-rasterization
submodule) -- skipped automatically otherwise.

IMPORTANT finding baked into this test's skip conditions: `slam.py`
unconditionally forces `use_gui = True` whenever `Dataset.type` is
`realsense`/`ROS` (see `SLAM.__init__` in `MonoGS/slam.py`), so live mode
*always* spawns the Open3D/GLFW GUI process -- there is currently no way to
run live mode fully headless. That process needs a working GL context, so
this test also requires a DISPLAY (e.g. via `xvfb-run` in CI/headless
environments) and will skip if one isn't available. This is a real
constraint on how tier-B system tests can be automated in CI, not a
limitation of this test file -- see the MANUAL_GUI_CHECKLIST.md and the
system test strategy plan for context.

This test is intentionally not run as part of the default `pytest` suite --
it belongs on a GPU-equipped (and, given the above, display-equipped/xvfb)
runner, invoked explicitly, e.g.:

    xvfb-run -a pytest MonoGS/tests/system/test_live_pipeline.py -v
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import rclpy
import yaml
from nav_msgs.msg import Path as PathMsg
from sensor_msgs.msg import PointCloud2

MONOGS_ROOT = Path(__file__).resolve().parents[2]


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _has_display():
    return bool(os.environ.get("DISPLAY")) or shutil.which("xvfb-run") is not None


def _has_video_publisher():
    return shutil.which("ros2") is not None


pytestmark = [
    pytest.mark.skipif(not _has_cuda(), reason="requires a CUDA GPU + built diff-gaussian-rasterization"),
    pytest.mark.skipif(not _has_display(), reason="live mode always spawns a GUI process; needs DISPLAY or xvfb-run"),
    pytest.mark.skipif(not _has_video_publisher(), reason="requires the ros_ws video_publisher package (ros2 run)"),
]


def _make_test_clip(tmp_dir):
    width, height, fps, n_frames = 64, 48, 10.0, 30
    video_path = tmp_dir / "clip.mp4"
    yaml_path = tmp_dir / "clip.yaml"

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 5) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    calibration = {
        "Dataset": {
            "Calibration": {
                "fx": 500.0,
                "fy": 500.0,
                "cx": width / 2.0,
                "cy": height / 2.0,
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
                "width": width,
                "height": height,
                "distorted": False,
            }
        }
    }
    yaml_path.write_text(yaml.safe_dump(calibration))
    return video_path, yaml_path


def _make_live_config(tmp_dir):
    with open(MONOGS_ROOT / "configs" / "live" / "ROS.yaml") as f:
        config = yaml.safe_load(f)
    config["Results"]["save_results"] = False
    config["Results"]["use_wandb"] = False
    config_path = tmp_dir / "test_live.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def test_live_pipeline_produces_trajectory_and_cloud(tmp_dir):
    video_path, calib_yaml = _make_test_clip(tmp_dir)
    config_path = _make_live_config(tmp_dir)

    publisher_proc = subprocess.Popen(
        [
            "ros2", "run", "video_publisher", "camera",
            "--ros-args",
            "-p", f"video_file:={video_path}",
            "-p", f"yaml_file:={calib_yaml}",
            "-p", "publish_rate:=10.0",
            "-p", "loop_video:=true",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    slam_proc = subprocess.Popen(
        [sys.executable, "slam.py", "--config", str(config_path)],
        cwd=str(MONOGS_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        rclpy.init()
        node = rclpy.create_node("live_pipeline_system_test")
        received = {"trajectory": None, "cloud": None}
        node.create_subscription(PathMsg, "/monoGS/trajectory", lambda msg: received.__setitem__("trajectory", msg), 10)
        node.create_subscription(PointCloud2, "/monoGS/cloud", lambda msg: received.__setitem__("cloud", msg), 10)

        deadline = time.time() + 120.0
        while time.time() < deadline and (received["trajectory"] is None or received["cloud"] is None):
            assert slam_proc.poll() is None, "slam.py exited unexpectedly:\n" + slam_proc.stdout.read().decode(errors="replace")
            assert publisher_proc.poll() is None, "video_publisher exited unexpectedly"
            rclpy.spin_once(node, timeout_sec=0.5)

        assert received["trajectory"] is not None, "No message received on /monoGS/trajectory"
        assert received["cloud"] is not None, "No message received on /monoGS/cloud"
        assert len(received["trajectory"].poses) >= 1
        assert received["cloud"].width * received["cloud"].height >= 1

        node.destroy_node()
        rclpy.shutdown()
    finally:
        for proc in (slam_proc, publisher_proc):
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in (slam_proc, publisher_proc):
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
                pytest.fail(f"process {proc.args[0]} did not exit cleanly after SIGINT and had to be killed")
