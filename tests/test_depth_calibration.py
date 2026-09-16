import numpy as np

from dlkit.utils.depth_calibration import median_scale_align


def test_median_scale_align():
    prediction = np.ones((8, 8), dtype=np.float32) * 0.5
    ground_truth = np.ones((8, 8), dtype=np.float32) * 2.0

    aligned = median_scale_align(prediction, ground_truth)

    assert abs(float(aligned.mean()) - 2.0) < 1e-3


def test_median_scale_align_ignores_all_invalid_ground_truth():
    prediction = np.ones((4, 4), dtype=np.float32)
    ground_truth = np.zeros((4, 4), dtype=np.float32)

    aligned = median_scale_align(prediction, ground_truth)

    assert float(aligned.mean()) == 1.0
