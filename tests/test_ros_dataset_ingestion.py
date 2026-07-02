"""data ingestion tests for utils.dataset.ROSDataset: topic wiring
and intrinsics resolution. Runs entirely on CPU -- ROSDataset.__init__ never
touches CUDA (only __getitem__ does, covered by test_ros_dataset_transform.py).
"""
import threading
import time

import numpy as np
import pytest
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image as ROSImage

from utils.dataset import ROSDataset


@pytest.fixture(autouse=True)
def _rclpy_context():
    # Function-scoped (not module-scoped): rclpy.spin_once() uses a process-wide
    # global executor that only resets on rclpy.shutdown(). Reusing it across
    # tests with different Node instances leaves stale generator state behind
    # ("ValueError: generator already executing") -- a fresh context per test
    # avoids that.
    rclpy.init()
    yield
    rclpy.shutdown()


def _base_config(camera_info_topic="/camera_info", depth_topic="None"):
    return {
        "Dataset": {
            "sensor_type": "monocular",
        },
        "ROS_topics": {
            "camera_topic": "/video_stream",
            "camera_info_topic": camera_info_topic,
            "depth_topic": depth_topic,
            "depth_scale": 1,
        },
    }


def _publish_in_background(node, image_pub, info_pub, bridge, width, height, stop_event):
    def _loop():
        frame = np.full((height, width, 3), 100, dtype=np.uint8)
        img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        info_msg = CameraInfo()
        info_msg.width = width
        info_msg.height = height
        info_msg.k = [500.0, 0.0, width / 2.0, 0.0, 500.0, height / 2.0, 0.0, 0.0, 1.0]
        info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        while not stop_event.is_set():
            image_pub.publish(img_msg)
            info_pub.publish(info_msg)
            time.sleep(0.05)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


def test_resolves_intrinsics_from_camera_info_topic():
    node = rclpy.create_node("ingestion_test_node_1")
    pub_node = rclpy.create_node("ingestion_test_publisher_1")
    bridge = CvBridge()

    image_pub = pub_node.create_publisher(ROSImage, "/video_stream", 10)
    info_pub = pub_node.create_publisher(CameraInfo, "/camera_info", 10)

    stop_event = threading.Event()
    pub_thread = _publish_in_background(pub_node, image_pub, info_pub, bridge, 64, 48, stop_event)

    try:
        dataset = ROSDataset(args=None, path="", config=_base_config(), node=node)
        assert dataset.fx == pytest.approx(500.0)
        assert dataset.fy == pytest.approx(500.0)
        assert dataset.cx == pytest.approx(32.0)
        assert dataset.cy == pytest.approx(24.0)
        assert dataset.width == 64
        assert dataset.height == 48
        assert dataset.has_depth is False
    finally:
        stop_event.set()
        pub_thread.join(timeout=2)
        node.destroy_node()
        pub_node.destroy_node()


def test_falls_back_to_config_calibration_when_no_camera_info_topic():
    node = rclpy.create_node("ingestion_test_node_2")
    pub_node = rclpy.create_node("ingestion_test_publisher_2")
    bridge = CvBridge()

    image_pub = pub_node.create_publisher(ROSImage, "/video_stream", 10)
    info_pub = pub_node.create_publisher(CameraInfo, "/camera_info", 10)  # unused by dataset

    stop_event = threading.Event()
    pub_thread = _publish_in_background(pub_node, image_pub, info_pub, bridge, 64, 48, stop_event)

    config = _base_config(camera_info_topic="None")
    config["Dataset"]["Calibration"] = {
        "fx": 111.0,
        "fy": 222.0,
        "cx": 33.0,
        "cy": 44.0,
        "width": 64,
        "height": 48,
        "distorted": False,
    }

    try:
        dataset = ROSDataset(args=None, path="", config=config, node=node)
        assert dataset.fx == 111.0
        assert dataset.fy == 222.0
        assert dataset.cx == 33.0
        assert dataset.cy == 44.0
    finally:
        stop_event.set()
        pub_thread.join(timeout=2)
        node.destroy_node()
        pub_node.destroy_node()


def test_depth_topic_none_subscribes_monocular_only():
    node = rclpy.create_node("ingestion_test_node_3")
    pub_node = rclpy.create_node("ingestion_test_publisher_3")
    bridge = CvBridge()

    image_pub = pub_node.create_publisher(ROSImage, "/video_stream", 10)
    info_pub = pub_node.create_publisher(CameraInfo, "/camera_info", 10)

    stop_event = threading.Event()
    pub_thread = _publish_in_background(pub_node, image_pub, info_pub, bridge, 64, 48, stop_event)

    try:
        dataset = ROSDataset(args=None, path="", config=_base_config(depth_topic="None"), node=node)
        assert not hasattr(dataset, "depth_sub")
        assert hasattr(dataset, "image_sub")
    finally:
        stop_event.set()
        pub_thread.join(timeout=2)
        node.destroy_node()
        pub_node.destroy_node()
