import numpy as np
import pytest

from semantic_perception.inference.crop_embeddings import (
    encode_object_crops,
    fuse_embeddings,
    make_crops,
)

LABEL = np.array([0.0, 1.0], dtype=np.float32)


def test_batches_and_normalizes_all_embedding_types():
    rgb = np.full((8, 8, 3), 127, dtype=np.uint8)
    boxes = [[0, 0, 4, 4], [4, 4, 8, 8]]
    masks = [np.ones((8, 8), dtype=bool), np.eye(8, dtype=bool)]
    batch_sizes = []

    def encoder(crops):
        batch_sizes.append(len(crops))
        return np.tile(np.array([[3.0, 4.0]], np.float32), (len(crops), 1))

    output = encode_object_crops(rgb, boxes, masks, [LABEL, LABEL], encoder)

    # One batch for the bbox crops and one for the masked crops.
    assert batch_sizes == [2, 2]
    for item in output:
        assert item.bbox_embedding.shape == item.mask_embedding.shape == (2,)
        assert item.label_embedding.shape == (2,)
        # concat of three D-dimensional components.
        assert item.fused_embedding.shape == (6,)
        assert np.linalg.norm(item.fused_embedding) == pytest.approx(1.0)


def test_concatenation_order_is_mask_then_bbox_then_label():
    mask = np.array([1.0, 0.0], dtype=np.float32)
    bbox = np.array([0.0, 1.0], dtype=np.float32)
    label = np.array([0.0, -1.0], dtype=np.float32)

    fused = fuse_embeddings(mask, bbox, label)

    expected = np.concatenate([mask, bbox, label])
    np.testing.assert_allclose(fused, expected / np.linalg.norm(expected), atol=1e-6)


def test_fused_dimension_is_three_times_the_component_dimension():
    component = np.ones(768, dtype=np.float32)
    fused = fuse_embeddings(component, component, component)
    assert fused.shape == (3 * 768,)
    assert np.linalg.norm(fused) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "mask,bbox,label",
    [
        (None, np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([1.0, 0.0]), None, np.array([0.0, 1.0])),
        (np.array([1.0, 0.0]), np.array([0.0, 1.0]), None),
        (np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([np.nan, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        # Dimensionally inconsistent components.
        (np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0])),
    ],
)
def test_invalid_components_reject_the_detection(mask, bbox, label):
    assert fuse_embeddings(mask, bbox, label).size == 0


def test_missing_label_embedding_rejects_the_detection():
    rgb = np.full((8, 8, 3), 127, dtype=np.uint8)
    warnings = []

    def encoder(crops):
        return np.tile(np.array([[3.0, 4.0]], np.float32), (len(crops), 1))

    output = encode_object_crops(
        rgb, [[0, 0, 4, 4]], [np.ones((8, 8), dtype=bool)], [None], encoder,
        warnings.append,
    )

    assert output[0].fused_embedding.size == 0
    assert not output[0].valid
    assert any("incomplete embedding components" in message for message in warnings)


def test_masked_crop_blacks_out_the_background():
    rgb = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True

    bbox_crop, masked_crop = make_crops(rgb, [0, 0, 4, 4], mask)

    # The bbox crop keeps the original pixels...
    assert (bbox_crop == 200).all()
    # ...while the masked crop keeps object pixels and zeroes everything else.
    assert (masked_crop[1:3, 1:3] == 200).all()
    assert masked_crop.sum() == masked_crop[1:3, 1:3].sum()
    assert masked_crop[0, 0].tolist() == [0, 0, 0]


def test_bbox_crop_uses_the_detector_box():
    rgb = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    bbox_crop, _ = make_crops(rgb, [2, 1, 6, 5], None)
    assert bbox_crop.shape == (4, 4, 3)
    np.testing.assert_array_equal(bbox_crop, rgb[1:5, 2:6])


def test_degenerate_boxes_produce_no_crop():
    rgb = np.full((8, 8, 3), 127, dtype=np.uint8)
    assert make_crops(rgb, [4, 4, 4, 4], None) == (None, None)
    assert make_crops(rgb, [0, 0, np.nan, 4], None) == (None, None)


def test_encoder_failure_propagates_unmodified():
    """No fatal/recoverable classification: every encoder failure propagates as-is."""
    rgb = np.full((8, 8, 3), 127, dtype=np.uint8)
    boxes = [[0, 0, 4, 4]]
    masks = [np.ones((8, 8), dtype=bool)]
    warnings = []
    calls = []

    def fail(crops):
        calls.append(len(crops))
        raise RuntimeError("CUDA error: unspecified launch failure")

    with pytest.raises(RuntimeError, match="unspecified launch failure"):
        encode_object_crops(rgb, boxes, masks, [LABEL], fail, warnings.append)
    assert calls == [1]
    assert warnings == []


def test_component_count_mismatch_is_rejected():
    rgb = np.full((8, 8, 3), 127, dtype=np.uint8)
    with pytest.raises(ValueError):
        encode_object_crops(rgb, [[0, 0, 4, 4]], [None], [], lambda crops: crops)
