from pathlib import Path

import numpy as np
import pytest

from semantic_perception.standalone import (
    intrinsics_from_horizontal_fov,
    load_sample,
    optical_camera_to_world,
)


def test_provided_rgb_depth_pose_sample_loads_and_is_synchronized():
    root = Path(__file__).parent
    rgb, depth, pose = load_sample(
        root / "Dd4bFSTQ8gi_000018_rgb.png",
        root / "Dd4bFSTQ8gi_000018_depth.png",
        root / "Dd4bFSTQ8gi_000018.txt",
    )
    assert rgb.shape == (720, 1080, 3)
    assert depth.shape == rgb.shape[:2]
    assert rgb.dtype == np.uint8 and depth.dtype == np.float32
    assert np.isfinite(pose).all() and pose.shape == (4, 4)
    assert pose[3].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert 0.0 < depth[depth > 0].min() <= depth.max() < 10.0
    assert intrinsics_from_horizontal_fov(1080, 720, 90.0) == pytest.approx(
        (540.0, 540.0, 539.5, 359.5)
    )
    np.testing.assert_allclose(
        optical_camera_to_world(np.eye(4), "habitat"),
        np.diag([1.0, -1.0, -1.0, 1.0]),
    )
