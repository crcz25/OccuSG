import numpy as np
import pytest

from semantic_perception.inference.crop_embeddings import encode_object_crops, fuse_embeddings


def test_batches_and_normalizes_all_embedding_types():
    rgb = np.full((8, 8, 3), 127, dtype=np.uint8)
    boxes = [[0, 0, 4, 4], [4, 4, 8, 8]]
    masks = [np.ones((8, 8), dtype=bool), np.eye(8, dtype=bool)]
    batch_sizes = []

    def encoder(crops):
        batch_sizes.append(len(crops))
        return np.tile(np.array([[3.0, 4.0]], np.float32), (len(crops), 1))

    output = encode_object_crops(rgb, boxes, masks, encoder, 0.5, 0.5)
    assert batch_sizes == [2, 2]
    for item in output:
        assert item.bbox_embedding.shape == item.mask_embedding.shape == item.fused_embedding.shape
        assert np.linalg.norm(item.fused_embedding) == pytest.approx(1.0)


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
        encode_object_crops(rgb, boxes, masks, fail, 0.5, 0.5, warnings.append)
    assert calls == [1]
    assert warnings == []


def test_fusion_falls_back_and_rejects_bad_weights():
    vector = np.array([3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(fuse_embeddings(vector, None, 0.5, 0.5), [0.6, 0.8])
    assert fuse_embeddings(None, None, 0.5, 0.5).size == 0
    with pytest.raises(ValueError):
        fuse_embeddings(vector, vector, 0.0, 0.0)
