"""transformation tests for ROSDataset.__getitem__: undistortion,
BGR->RGB colorspace conversion, and normalization to a [0,1] CHW tensor.

ROSDataset.__init__ requires a live ROS graph and blocks until intrinsics
arrive (see test_ros_dataset_ingestion.py), which is overkill for testing
pure transform logic. Instead we build the object with object.__new__(),
set only the attributes __getitem__ actually reads, and pin the device to
CPU -- this isolates the transform math from ROS wiring and from the
"cuda:0" device in BaseDataset.__init__, keeping this test GPU-free.
"""
import cv2
import numpy as np
import pytest
import torch

from utils.dataset import ROSDataset


def _make_dataset(image_bgr, distorted):
    dataset = object.__new__(ROSDataset)
    dataset.device = "cpu"
    dataset.dtype = torch.float32
    dataset.image = image_bgr
    dataset.depth = None
    dataset.disorted = distorted
    dataset.config = {
        "ROS_topics": {"depth_topic": "None"},
    }

    if distorted:
        height, width = image_bgr.shape[:2]
        K = np.array([[80.0, 0.0, width / 2.0], [0.0, 80.0, height / 2.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.array([0.15, -0.05, 0.0, 0.0, 0.0])
        dataset.map1x, dataset.map1y = cv2.initUndistortRectifyMap(
            K, dist_coeffs, np.eye(3), K, (width, height), cv2.CV_32FC1
        )
    else:
        dataset.map1x, dataset.map1y = None, None

    return dataset


def _sample_bgr_image(height=48, width=64):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = 10   # B
    image[:, :, 1] = 20   # G
    image[:, :, 2] = 200  # R
    return image


def test_output_shape_dtype_and_value_range():
    image_bgr = _sample_bgr_image()
    dataset = _make_dataset(image_bgr, distorted=False)

    tensor, depth, pose = dataset[0]

    assert tensor.shape == (3, 48, 64)  # CHW
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    assert depth is None
    assert torch.equal(pose, torch.eye(4, dtype=torch.float32))


def test_bgr_converted_to_rgb_channel_order():
    image_bgr = _sample_bgr_image()
    dataset = _make_dataset(image_bgr, distorted=False)

    tensor, _, _ = dataset[0]

    # Channel 0 of the CHW tensor should now be R (~200/255), channel 2 should be B (~10/255)
    assert tensor[0].mean().item() == pytest.approx(200 / 255.0, abs=1e-3)
    assert tensor[2].mean().item() == pytest.approx(10 / 255.0, abs=1e-3)


def test_undistortion_is_applied_when_distorted():
    image_bgr = _sample_bgr_image()
    # Put a distinct marker pixel so we can tell undistortion moved content around.
    image_bgr[5:8, 5:8, :] = np.array([255, 255, 255], dtype=np.uint8)

    dataset_distorted = _make_dataset(image_bgr.copy(), distorted=True)
    dataset_plain = _make_dataset(image_bgr.copy(), distorted=False)

    tensor_distorted, _, _ = dataset_distorted[0]
    tensor_plain, _, _ = dataset_plain[0]

    assert not torch.equal(tensor_distorted, tensor_plain)


def test_depth_topic_none_yields_no_depth():
    image_bgr = _sample_bgr_image()
    dataset = _make_dataset(image_bgr, distorted=False)

    _, depth, _ = dataset[0]

    assert depth is None
