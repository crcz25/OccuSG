import numpy as np

from semantic_perception.inference.geometry import compute_geometry


def test_geometry_and_invalid_depth():
    depth = np.ones((4, 4), dtype=np.float32) * 2.0
    mask = np.ones((4, 4), dtype=bool)
    result = compute_geometry(depth, mask, [0, 0, 4, 4], [2, 2, 1.5, 1.5], 4, 5.0)
    assert result.valid
    np.testing.assert_allclose(result.centroid, [0, 0, 2], atol=1e-6)

    depth[:] = np.nan
    assert not compute_geometry(depth, mask, [0, 0, 4, 4], [2, 2, 1.5, 1.5], 4).valid
