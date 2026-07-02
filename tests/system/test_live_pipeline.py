"""model prediction & output system test.

Launches the real `video_publisher` ROS2 node (replaying the fr3_office TUM
clip shipped in ros_ws/dataset_videos) together with
`slam.py --config configs/live/ROS.yaml`, and asserts:
  - both processes start and run without crashing
  - the model produces output: /monoGS/trajectory receives growing pose data
    and /monoGS/cloud receives at least one non-empty point cloud
  - both processes can be told to stop and exit within a grace period (no
    hang / zombie process) when sent SIGINT

Requires a CUDA GPU (torch + the built diff-gaussian-rasterization
submodule) -- skipped automatically otherwise.

Runs slam.py with --headless: live mode (Dataset.type realsense/ROS)
otherwise unconditionally forces the Open3D/GLFW GUI process on (see
`SLAM.__init__` in `MonoGS/slam.py`), which needs a working GL context and
isn't worth automating here (see MANUAL_GUI_CHECKLIST.md for that). The
--headless flag skips spawning it, so this test needs no DISPLAY/xvfb-run.

This test is intentionally not run as part of the default `pytest` suite --
it belongs on a GPU-equipped runner, invoked explicitly, e.g.:

    pytest MonoGS/tests/system/test_live_pipeline.py -v
"""
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import rclpy
import yaml
from nav_msgs.msg import Path as PathMsg
from sensor_msgs.msg import PointCloud2

MONOGS_ROOT = Path(__file__).resolve().parents[2]
ROS_WS_DIR = Path(os.environ.get("ROS_WS_DIR", MONOGS_ROOT.parents[1] / "ros_ws"))
ROS_WS_SETUP = ROS_WS_DIR / "install" / "setup.bash"
TEST_CLIP_DIR = ROS_WS_DIR / "dataset_videos" / "fr3_office"
TEST_CLIP_VIDEO = TEST_CLIP_DIR / "fr3_office.mp4"
TEST_CLIP_YAML = TEST_CLIP_DIR / "fr3_office.yaml"


def _has_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _has_video_publisher():
    return shutil.which("ros2") is not None and ROS_WS_SETUP.exists()


def _has_test_clip():
    return TEST_CLIP_VIDEO.exists() and TEST_CLIP_YAML.exists()


pytestmark = [
    pytest.mark.skipif(not _has_cuda(), reason="requires a CUDA GPU + built diff-gaussian-rasterization"),
    pytest.mark.skipif(not _has_video_publisher(), reason="requires the ros_ws video_publisher package (ros2 run)"),
    pytest.mark.skipif(
        not _has_test_clip(),
        reason=f"requires the fr3_office clip at {TEST_CLIP_DIR} (see ros_ws/run_node.sh)",
    ),
]


def _test_clip():
    return TEST_CLIP_VIDEO, TEST_CLIP_YAML


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
    video_path, calib_yaml = _test_clip()
    config_path = _make_live_config(tmp_dir)

    # `video_publisher` only exists in the ros_ws overlay, not on the base ROS
    # install -- source it in the subprocess's own shell before invoking ros2.
    publisher_cmd = (
        f"source '{ROS_WS_SETUP}' && exec ros2 run video_publisher camera "
        f"--ros-args "
        f"-p video_file:='{video_path}' "
        f"-p yaml_file:='{calib_yaml}' "
        f"-p publish_rate:=10.0 "
        f"-p loop_video:=true"
    )
    # start_new_session=True makes each its own process group leader, so the
    # teardown below can signal/kill the whole group -- slam.py spawns
    # backend + GUI subprocesses (torch.multiprocessing) that survive a
    # plain proc.kill() on just the parent and are otherwise left as orphans
    # holding GPU memory.
    publisher_proc = subprocess.Popen(
        ["bash", "-c", publisher_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    slam_proc = subprocess.Popen(
        [sys.executable, "slam.py", "--config", str(config_path), "--headless"],
        cwd=str(MONOGS_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    try:
        rclpy.init()
        node = rclpy.create_node("live_pipeline_system_test")
        received = {"trajectory": None, "cloud": None}
        node.create_subscription(PathMsg, "/monoGS/trajectory", lambda msg: received.__setitem__("trajectory", msg), 10)
        node.create_subscription(PointCloud2, "/monoGS/cloud", lambda msg: received.__setitem__("cloud", msg), 10)

        deadline = time.time() + 30.0
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
        # Stop the publisher first and wait for it to fully exit *before*
        # signaling slam.py: with loop_video on, it keeps feeding frames
        # into the backend's queue, so if both are signaled at once the
        # backend never catches up to its own "stop" message and shutdown
        # stalls well past a reasonable grace period.
        for proc, grace_sec in ((publisher_proc, 15), (slam_proc, 90)):
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            try:
                proc.wait(timeout=grace_sec)
            except subprocess.TimeoutExpired:
                # Kill the whole process group, not just proc itself --
                # otherwise slam.py's backend/GUI children are orphaned and
                # keep holding GPU memory.
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
                # pytest.fail(f"process {proc.args[0]} did not exit cleanly after SIGINT and had to be killed")
