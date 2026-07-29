import numpy as np

from semantic_perception.debug_image import render_debug_image
from semantic_perception.inference.crop_embeddings import CropEmbedding
from semantic_perception.inference.geometry import Geometry3D
from semantic_perception.inference.models import Detection
from semantic_perception.worker import Proposal


def test_debug_image_contains_mask_box_and_class_label():
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    mask = np.zeros((80, 100), dtype=bool)
    mask[20:60, 25:75] = True
    embedding = CropEmbedding(
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )
    proposal = Proposal(
        Detection(
            np.array([20, 15, 80, 65], dtype=np.float32),
            0.875,
            class_index=0,
            class_name="chair",
        ),
        mask,
        embedding,
        Geometry3D.invalid(),
    )

    rendered = render_debug_image(
        rgb, [proposal], lambda _: ("chair", 0.9), mask_alpha=0.5
    )

    assert rendered.shape == rgb.shape
    assert rendered.dtype == np.uint8
    assert rendered[40, 50].any()  # Mask overlay.
    assert rendered[15, 20].any()  # Bounding box.

