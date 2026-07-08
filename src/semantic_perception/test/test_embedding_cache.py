import numpy as np

from semantic_perception.inference.embedding_cache import HEADER, MAGIC, VERSION, load_or_generate


def test_cache_generated_reused_and_invalidated(tmp_path):
    csv_path = tmp_path / "classes.csv"
    cache_path = tmp_path / "cache.bin"
    csv_path.write_text("chair\ntable\n", encoding="utf-8")
    calls = []

    def encoder(prompts):
        calls.append(list(prompts))
        return np.arange(1, len(prompts) * 3 + 1, dtype=np.float32).reshape(-1, 3)

    prompts, first, reused = load_or_generate(csv_path, cache_path, "ViT-X", encoder)
    assert prompts == ["chair", "table"]
    assert not reused
    assert len(calls) == 1
    magic, version, count, dimension, _ = HEADER.unpack_from(cache_path.read_bytes())
    assert (magic, version, count, dimension) == (MAGIC, VERSION, 2, 3)

    _, second, reused = load_or_generate(csv_path, cache_path, "ViT-X", encoder)
    assert reused
    assert len(calls) == 1
    np.testing.assert_array_equal(first, second)

    csv_path.write_text("chair\ntable\nlamp\n", encoding="utf-8")
    _, third, reused = load_or_generate(csv_path, cache_path, "ViT-X", encoder)
    assert not reused
    assert len(calls) == 2
    assert third.shape == (3, 3)


def test_bad_cache_is_regenerated(tmp_path):
    csv_path = tmp_path / "classes.csv"
    cache_path = tmp_path / "cache.bin"
    csv_path.write_text("chair\n", encoding="utf-8")
    cache_path.write_bytes(b"bad")
    _, values, reused = load_or_generate(
        csv_path, cache_path, "model", lambda _: np.array([[3.0, 4.0]], np.float32)
    )
    assert not reused
    np.testing.assert_allclose(values, [[0.6, 0.8]])


def test_hm3d_count_table_extracts_only_labels(tmp_path):
    csv_path = tmp_path / "hm3d.csv"
    cache_path = tmp_path / "cache.bin"
    csv_path.write_text(
        "Object Type Name; # of instances in semantic text files\n"
        "wall;25036\nchair;2339\n",
        encoding="utf-8",
    )
    observed = []

    def encoder(prompts):
        observed.extend(prompts)
        return np.ones((len(prompts), 2), dtype=np.float32)

    prompts, matrix, reused = load_or_generate(
        csv_path, cache_path, "ViT-H-14", encoder
    )
    assert prompts == observed == ["wall", "chair"]
    assert matrix.shape == (2, 2)
    assert not reused
