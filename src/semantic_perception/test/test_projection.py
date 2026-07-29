import numpy as np
import pytest

from semantic_perception.inference.geometry import compute_geometry
from semantic_perception.projection import (
    ProjectionDiagnostics,
    depth_statistics,
    quaternion_to_rotation_matrix,
    transform_to_matrix,
)


class _Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x, self.y, self.z, self.w = x, y, z, w


class _Transform:
    def __init__(self, translation, rotation):
        self.translation = translation
        self.rotation = rotation


class _Stamped:
    def __init__(self, translation, rotation):
        self.transform = _Transform(translation, rotation)


# Pinhole intrinsics matching the validation bag: 640x480, f=320, principal point centred.
INTRINSICS = (320.0, 320.0, 320.0, 240.0)


def _depth_plane(value=2.0, shape=(480, 640)):
    return np.full(shape, value, dtype=np.float32)


def _centre_mask(shape=(480, 640), half=20):
    mask = np.zeros(shape, dtype=bool)
    mask[240 - half : 240 + half, 320 - half : 320 + half] = True
    return mask


# ========== transform conversion ==========


def test_identity_transform_is_the_identity_matrix():
    matrix = transform_to_matrix(_Stamped(_Vector(), _Vector()))
    np.testing.assert_allclose(matrix, np.eye(4), atol=1e-9)


def test_transform_applies_translation_in_the_target_frame():
    matrix = transform_to_matrix(_Stamped(_Vector(1.0, 2.0, 3.0), _Vector()))
    point = matrix @ np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(point[:3], [1.0, 2.0, 3.0], atol=1e-9)


def test_quaternion_rotation_matches_a_known_yaw():
    # 90 degrees about +Z maps +X onto +Y.
    rotation = quaternion_to_rotation_matrix(0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-9)


def test_transform_normalizes_an_unnormalized_quaternion():
    matrix = transform_to_matrix(_Stamped(_Vector(), _Vector(0.0, 0.0, 0.0, 2.0)))
    np.testing.assert_allclose(matrix[:3, :3], np.eye(3), atol=1e-9)


@pytest.mark.parametrize(
    "stamped",
    [
        None,
        _Stamped(_Vector(float("nan")), _Vector()),
        _Stamped(_Vector(), _Vector(0.0, 0.0, 0.0, 0.0)),
        _Stamped(_Vector(), _Vector(float("inf"), 0.0, 0.0, 1.0)),
    ],
)
def test_invalid_transforms_are_rejected(stamped):
    assert transform_to_matrix(stamped) is None


# ========== camera to graph projection ==========


def test_projection_places_a_centred_object_on_the_optical_axis():
    geometry = compute_geometry(
        _depth_plane(2.0), _centre_mask(), [300, 220, 340, 260], INTRINSICS
    )
    assert geometry.valid
    # A patch centred on the principal point projects to (0, 0, depth), up to the
    # half-pixel offset between the pixel-centre grid and the principal point.
    np.testing.assert_allclose(geometry.centroid, [0.0, 0.0, 2.0], atol=1e-2)


def test_projection_uses_optical_frame_axes():
    # A patch right of and below the principal point yields +x and +y in the
    # optical convention (x right, y down, z forward).
    mask = np.zeros((480, 640), dtype=bool)
    mask[300:340, 400:440] = True
    geometry = compute_geometry(
        _depth_plane(2.0), mask, [400, 300, 440, 340], INTRINSICS
    )
    assert geometry.valid
    assert geometry.centroid[0] > 0.0
    assert geometry.centroid[1] > 0.0
    assert geometry.centroid[2] == pytest.approx(2.0, abs=1e-3)


def test_camera_to_world_transform_is_applied():
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = [10.0, -5.0, 1.0]

    geometry = compute_geometry(
        _depth_plane(2.0),
        _centre_mask(),
        [300, 220, 340, 260],
        INTRINSICS,
        camera_to_world=camera_to_world,
    )

    assert geometry.valid
    np.testing.assert_allclose(geometry.centroid, [10.0, -5.0, 3.0], atol=1e-2)


def test_rotated_transform_matches_manual_composition():
    rotation = quaternion_to_rotation_matrix(0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = rotation
    camera_to_world[:3, 3] = [1.0, 2.0, 3.0]

    camera_only = compute_geometry(
        _depth_plane(2.0), _centre_mask(), [300, 220, 340, 260], INTRINSICS
    )
    world = compute_geometry(
        _depth_plane(2.0),
        _centre_mask(),
        [300, 220, 340, 260],
        INTRINSICS,
        camera_to_world=camera_to_world,
    )

    expected = rotation @ np.asarray(camera_only.centroid, dtype=np.float64) + [1.0, 2.0, 3.0]
    np.testing.assert_allclose(world.centroid, expected, atol=1e-4)


def test_invalid_transform_matrix_rejects_the_detection():
    broken = np.full((4, 4), np.nan)
    geometry = compute_geometry(
        _depth_plane(), _centre_mask(), [300, 220, 340, 260], INTRINSICS,
        camera_to_world=broken,
    )
    assert not geometry.valid


# ========== depth handling ==========


def test_depth_is_interpreted_in_metres():
    near = compute_geometry(_depth_plane(1.0), _centre_mask(), [300, 220, 340, 260], INTRINSICS)
    far = compute_geometry(_depth_plane(4.0), _centre_mask(), [300, 220, 340, 260], INTRINSICS)
    assert near.centroid[2] == pytest.approx(1.0, abs=1e-3)
    assert far.centroid[2] == pytest.approx(4.0, abs=1e-3)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_depth_values_are_excluded(value):
    depth = _depth_plane(value)
    geometry = compute_geometry(depth, _centre_mask(), [300, 220, 340, 260], INTRINSICS)
    assert not geometry.valid


def test_depth_beyond_the_maximum_is_excluded():
    geometry = compute_geometry(
        _depth_plane(50.0), _centre_mask(), [300, 220, 340, 260], INTRINSICS,
        max_depth_m=10.0,
    )
    assert not geometry.valid


def test_too_few_valid_points_rejects_the_detection():
    mask = np.zeros((480, 640), dtype=bool)
    mask[240:242, 320:322] = True  # 4 pixels
    geometry = compute_geometry(
        _depth_plane(2.0), mask, [318, 238, 324, 244], INTRINSICS, min_valid_points=20
    )
    assert not geometry.valid


def test_uncalibrated_intrinsics_reject_the_detection():
    for intrinsics in ((0.0, 320.0, 320.0, 240.0), (320.0, -1.0, 320.0, 240.0)):
        geometry = compute_geometry(
            _depth_plane(), _centre_mask(), [300, 220, 340, 260], intrinsics
        )
        assert not geometry.valid


def test_mask_shape_must_match_the_depth_image():
    geometry = compute_geometry(
        _depth_plane(), np.ones((10, 10), dtype=bool), [300, 220, 340, 260], INTRINSICS
    )
    assert not geometry.valid


def test_mask_restricts_projection_to_object_pixels():
    depth = _depth_plane(2.0)
    depth[:, 320:] = 8.0  # right half is far away
    mask = np.zeros((480, 640), dtype=bool)
    mask[220:260, 260:320] = True  # only the near (left) half

    geometry = compute_geometry(depth, mask, [200, 200, 440, 280], INTRINSICS)

    assert geometry.valid
    assert geometry.centroid[2] == pytest.approx(2.0, abs=1e-3)


# ========== diagnostics ==========


def test_depth_statistics_report_median_range_and_rejections():
    depth = _depth_plane(2.0)
    depth[240:245, 320:325] = 0.0  # 25 invalid pixels inside the mask
    mask = _centre_mask()

    median, minimum, maximum, rejected = depth_statistics(depth, mask, [280, 200, 360, 280], 10.0)

    assert median == pytest.approx(2.0)
    assert minimum == pytest.approx(2.0)
    assert maximum == pytest.approx(2.0)
    assert rejected == 25


def test_depth_statistics_handle_a_fully_invalid_selection():
    stats = depth_statistics(_depth_plane(0.0), _centre_mask(), [300, 220, 340, 260], 10.0)
    assert stats == (0.0, 0.0, 0.0, 40 * 40)


def test_diagnostics_render_frames_timestamps_and_positions():
    diagnostics = ProjectionDiagnostics(
        sequence=7,
        source_frame="rgb_camera_optical_frame",
        target_frame="odom",
        frame_stamp_sec=100.5,
        transform_stamp_sec=100.4,
        transform=np.eye(4),
    )
    diagnostics.add_detection(3, "chair", (0.1, 0.2, 2.0), (5.0, 6.0, 1.0), 812, (2.0, 1.8, 2.4, 12))

    text = diagnostics.render()

    assert "seq=7" in text
    assert "rgb_camera_optical_frame->odom" in text
    assert "frame_stamp=100.500000" in text
    assert "tf_stamp=100.400000" in text
    assert "id=3" in text and "'chair'" in text
    assert "camera=(0.100, 0.200, 2.000)" in text
    assert "graph=(5.000, 6.000, 1.000)" in text
    assert "mask_px=812" in text
    assert "depth median=2.000 min=1.800 max=2.400 rejected=12" in text
