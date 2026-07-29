import numpy as np
import pytest

from semantic_perception.inference.embedding_cache import build_label_embedding_cache
from semantic_perception.inference.models import (
    match_phrase_to_class,
    non_maximum_suppression,
)
from semantic_perception.inference.vocabulary import (
    DetectorPrompt,
    Vocabulary,
    VocabularyError,
    build_detector_prompt,
    canonical_label,
    load_class_labels,
    parse_excluded_labels,
    resolve_global_index,
)

HM3D_HEADER = "Object Type Name; # of instances in semantic text files"


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ========== CSV loading ==========


def test_loads_hm3d_table_and_ignores_the_header(tmp_path):
    path = _write(tmp_path, "hm3d.csv", f"{HM3D_HEADER}\nwall;25036\nchair;120\n")
    labels, source = load_class_labels(path)
    assert labels == ["wall", "chair"]
    assert source == path.read_bytes()


def test_loads_a_plain_one_column_csv(tmp_path):
    path = _write(tmp_path, "plain.csv", "chair\ntable\nsofa\n")
    assert load_class_labels(path)[0] == ["chair", "table", "sofa"]


def test_preserves_file_order(tmp_path):
    path = _write(tmp_path, "hm3d.csv", f"{HM3D_HEADER}\nwall;9\nchair;8\nlamp;7\n")
    assert load_class_labels(path)[0] == ["wall", "chair", "lamp"]


def test_skips_empty_rows(tmp_path):
    path = _write(tmp_path, "hm3d.csv", f"{HM3D_HEADER}\nwall;1\n\n  \nchair;2\n\n")
    assert load_class_labels(path)[0] == ["wall", "chair"]


def test_skips_duplicate_labels_keeping_the_first(tmp_path):
    path = _write(
        tmp_path, "hm3d.csv", f"{HM3D_HEADER}\nchair;5\nChair;3\n  chair ;1\ntable;2\n"
    )
    assert load_class_labels(path)[0] == ["chair", "table"]


def test_trims_whitespace_without_changing_the_label(tmp_path):
    path = _write(tmp_path, "hm3d.csv", f"{HM3D_HEADER}\n  potted plant  ;5\n")
    assert load_class_labels(path)[0] == ["potted plant"]


def test_malformed_rows_are_skipped_not_fatal(tmp_path):
    path = _write(tmp_path, "hm3d.csv", f"{HM3D_HEADER}\n;5\nchair;notanumber\ntable;2\n")
    # A blank label is skipped; a bad count does not invalidate a usable label.
    assert load_class_labels(path)[0] == ["chair", "table"]


def test_missing_file_fails_clearly(tmp_path):
    with pytest.raises(VocabularyError, match="does not exist"):
        load_class_labels(tmp_path / "absent.csv")


def test_file_without_valid_labels_fails_clearly(tmp_path):
    path = _write(tmp_path, "hm3d.csv", f"{HM3D_HEADER}\n\n  \n")
    with pytest.raises(VocabularyError, match="no valid labels"):
        load_class_labels(path)


def test_real_hm3d_table_loads(tmp_path):
    from pathlib import Path

    table = Path("/workspace/occusg_ws/models/labels/HM3D_CountsOfObjectTypes.csv")
    if not table.is_file():
        pytest.skip("HM3D vocabulary is not present in this checkout")
    labels, _ = load_class_labels(table)
    assert len(labels) > 1000
    assert labels[0] == "wall"
    assert len(set(canonical_label(label) for label in labels)) == len(labels)


# ========== index mapping ==========


def test_vocabulary_lookup_is_case_and_whitespace_insensitive():
    vocabulary = Vocabulary(("Chair", "potted plant"))
    assert vocabulary.index_of("chair") == 0
    assert vocabulary.index_of("  POTTED   PLANT ") == 1
    assert vocabulary.index_of("sofa") is None
    assert vocabulary.label_at(0) == "Chair"


def test_single_prompt_covers_every_label_exactly_once():
    vocabulary = Vocabulary(tuple(f"class{i}" for i in range(10)))
    prompt = vocabulary.build_prompt()

    assert prompt.labels == vocabulary.labels
    assert prompt.global_indices == tuple(range(10))
    for local, global_index in enumerate(prompt.global_indices):
        assert vocabulary.label_at(global_index) == prompt.labels[local]


def test_caption_uses_dot_separated_classes():
    prompt = Vocabulary(("chair", "table", "potted plant")).build_prompt()
    # The format Grounding DINO expects for multi-class captions.
    assert prompt.caption == "chair . table . potted plant ."


def test_caption_is_one_pass_for_the_whole_vocabulary():
    vocabulary = Vocabulary(tuple(f"class{i}" for i in range(64)))
    caption = vocabulary.build_prompt().caption
    assert caption.count(" . ") == 63
    assert caption.endswith(" .")
    for label in vocabulary.labels:
        assert label in caption


def test_prompt_maps_local_index_to_global_index():
    prompt = DetectorPrompt(labels=("a", "b"), global_indices=(8, 9))
    assert prompt.global_index(0) == 8
    assert prompt.global_index(1) == 9
    assert prompt.global_index(2) is None
    assert prompt.global_index(-1) is None


def test_prompt_construction_is_deterministic():
    vocabulary = Vocabulary(tuple(f"class{i}" for i in range(23)))
    assert vocabulary.build_prompt() == vocabulary.build_prompt()


def test_detector_prompt_keeps_the_leading_labels():
    vocabulary = Vocabulary(("a", "b", "c", "d"))
    prompt = build_detector_prompt(vocabulary, 2)
    assert prompt.labels == ("a", "b")
    assert prompt.global_indices == (0, 1)
    assert prompt.caption == "a . b ."

    whole = build_detector_prompt(vocabulary, 0)
    assert whole.labels == vocabulary.labels
    assert whole.caption == "a . b . c . d ."


# ========== prompt exclusions ==========


def test_excluded_labels_are_absent_from_the_caption():
    vocabulary = Vocabulary(("wall", "floor", "chair", "ceiling", "table"))
    prompt = build_detector_prompt(vocabulary, 0, ("floor", "ceiling"))

    assert prompt.labels == ("wall", "chair", "table")
    assert prompt.caption == "wall . chair . table ."
    assert "floor" not in prompt.caption
    assert "ceiling" not in prompt.caption


def test_exclusions_preserve_global_indices():
    vocabulary = Vocabulary(("wall", "floor", "chair", "ceiling", "table"))
    prompt = build_detector_prompt(vocabulary, 0, ("floor", "ceiling"))

    # The prompt is a non-contiguous subset, so the map must not be 0..N-1.
    assert prompt.global_indices == (0, 2, 4)
    for local, global_index in enumerate(prompt.global_indices):
        assert vocabulary.label_at(global_index) == prompt.labels[local]
    # And a detector result still resolves to the right vocabulary entry.
    assert resolve_global_index(prompt, vocabulary, 1, "chair") == 2
    assert vocabulary.label_at(2) == "chair"


def test_exclusions_apply_before_the_size_limit():
    vocabulary = Vocabulary(("wall", "floor", "chair", "ceiling", "table"))
    prompt = build_detector_prompt(vocabulary, 3, ("floor", "ceiling"))

    # Excluding two of the leading labels backfills rather than shrinking.
    assert prompt.labels == ("wall", "chair", "table")
    assert len(prompt.labels) == 3


def test_exclusions_keep_the_vocabulary_and_its_embeddings_intact():
    vocabulary = Vocabulary(("wall", "floor", "chair"))
    build_detector_prompt(vocabulary, 0, ("floor",))

    # The excluded label is still addressable, so its cached embedding stays valid.
    assert len(vocabulary) == 3
    assert vocabulary.index_of("floor") == 1
    assert vocabulary.label_at(1) == "floor"


def test_exclusions_are_case_and_whitespace_insensitive():
    vocabulary = Vocabulary(("Wall", "Floor", "potted plant"))
    prompt = build_detector_prompt(vocabulary, 0, ("  FLOOR ", "POTTED   PLANT"))
    assert prompt.labels == ("Wall",)


def test_unknown_exclusions_are_reported_and_harmless():
    vocabulary = Vocabulary(("wall", "floor"))
    assert vocabulary.unknown_labels(("floor", "spaceship")) == ("spaceship",)
    prompt = build_detector_prompt(vocabulary, 0, ("spaceship",))
    assert prompt.labels == ("wall", "floor")


def test_excluding_everything_is_an_error():
    vocabulary = Vocabulary(("wall", "floor"))
    with pytest.raises(VocabularyError, match="Every vocabulary label is excluded"):
        build_detector_prompt(vocabulary, 0, ("wall", "floor"))


def test_prompt_records_its_exclusions():
    vocabulary = Vocabulary(("wall", "floor"))
    prompt = build_detector_prompt(vocabulary, 0, ("floor",))
    assert prompt.excluded_labels == ("floor",)


# ========== exclusion parameter parsing ==========


def test_parses_a_comma_separated_string():
    assert parse_excluded_labels("floor, ceiling") == ("floor", "ceiling")


def test_parsing_trims_whitespace_and_drops_blanks():
    assert parse_excluded_labels(" floor ,, ceiling , ") == ("floor", "ceiling")
    assert parse_excluded_labels("  potted   plant ") == ("potted plant",)


def test_parsing_collapses_duplicates_keeping_order():
    assert parse_excluded_labels("floor, Floor, ceiling, floor") == ("floor", "ceiling")


def test_parsing_accepts_an_empty_or_missing_value():
    assert parse_excluded_labels("") == ()
    assert parse_excluded_labels("   ") == ()
    assert parse_excluded_labels(",,") == ()
    assert parse_excluded_labels(None) == ()


def test_parsing_accepts_an_already_split_sequence():
    assert parse_excluded_labels(["floor", " ceiling "]) == ("floor", "ceiling")


def test_configured_exclusions_apply_to_the_real_hm3d_vocabulary(tmp_path):
    from pathlib import Path

    table = Path("/workspace/occusg_ws/models/labels/HM3D_CountsOfObjectTypes.csv")
    if not table.is_file():
        pytest.skip("HM3D vocabulary is not present in this checkout")
    vocabulary = Vocabulary(tuple(load_class_labels(table)[0]))
    excluded = parse_excluded_labels("floor, ceiling")

    baseline = build_detector_prompt(vocabulary, 64)
    filtered = build_detector_prompt(vocabulary, 64, excluded)

    assert "ceiling" in baseline.labels
    assert "ceiling" not in filtered.labels
    assert "floor" not in filtered.labels
    # Still a full-size caption, backfilled from further down the vocabulary.
    assert len(filtered.labels) == len(baseline.labels) == 64
    for local, global_index in enumerate(filtered.global_indices):
        assert vocabulary.label_at(global_index) == filtered.labels[local]


def test_resolve_global_index_prefers_the_detector_index():
    vocabulary = Vocabulary(("chair", "table", "sofa"))
    prompt = DetectorPrompt(("table", "sofa"), (1, 2))
    assert resolve_global_index(prompt, vocabulary, 0, "table") == 1
    assert resolve_global_index(prompt, vocabulary, 1, "sofa") == 2


def test_resolve_global_index_falls_back_to_the_phrase():
    vocabulary = Vocabulary(("chair", "table"))
    prompt = DetectorPrompt(("chair", "table"), (0, 1))
    assert resolve_global_index(prompt, vocabulary, None, "table") == 1


def test_resolve_global_index_rejects_unknown_phrases():
    vocabulary = Vocabulary(("chair", "table"))
    prompt = DetectorPrompt(("chair", "table"), (0, 1))
    assert resolve_global_index(prompt, vocabulary, None, "spaceship") is None
    assert resolve_global_index(prompt, vocabulary, None, "") is None
    assert resolve_global_index(prompt, vocabulary, 7, "") is None


def test_merged_adjacent_phrase_maps_deterministically():
    # A single multi-class caption can return a phrase spanning two neighbouring
    # classes ("glass oil"); the phrase matcher resolves it before the index
    # lookup, so the mapping stays deterministic end to end.
    vocabulary = Vocabulary(("glass", "oil"))
    prompt = vocabulary.build_prompt()

    local = match_phrase_to_class("glass oil", prompt.labels)
    assert local == 0
    assert resolve_global_index(prompt, vocabulary, local, "glass oil") == 0
    assert vocabulary.label_at(0) == "glass"


def test_unmatchable_phrase_yields_no_index():
    vocabulary = Vocabulary(("glass", "oil"))
    prompt = vocabulary.build_prompt()
    local = match_phrase_to_class("bicycle", prompt.labels)
    assert local is None
    assert resolve_global_index(prompt, vocabulary, local, "bicycle") is None


def test_duplicate_labels_are_rejected_by_the_vocabulary():
    with pytest.raises(VocabularyError):
        Vocabulary(("chair", "Chair"))


# ========== phrase matching ==========


def test_phrase_matching_prefers_the_longest_class():
    # Upstream phrases2classes would return "chair" because it is listed first.
    classes = ["chair", "armchair"]
    assert match_phrase_to_class("armchair", classes) == 1
    assert match_phrase_to_class("chair", classes) == 0


def test_phrase_matching_is_case_insensitive():
    assert match_phrase_to_class("  Potted Plant ", ["potted plant"]) == 0


def test_phrase_matching_returns_none_for_unmatched_phrases():
    assert match_phrase_to_class("bicycle", ["chair", "table"]) is None
    assert match_phrase_to_class("", ["chair"]) is None


# ========== cross-batch merge ==========


def test_non_maximum_suppression_keeps_the_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.5, 0.9, 0.4])
    kept = non_maximum_suppression(boxes, scores, 0.5)
    assert kept == [1, 2]


def test_non_maximum_suppression_keeps_disjoint_boxes():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    kept = non_maximum_suppression(boxes, np.array([0.5, 0.4]), 0.5)
    assert sorted(kept) == [0, 1]


def test_non_maximum_suppression_is_deterministic_on_ties():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    assert non_maximum_suppression(boxes, np.array([0.5, 0.5]), 0.5) == [0]


# ========== label embedding cache ==========


def test_label_cache_encodes_every_label_once(tmp_path):
    path = _write(tmp_path, "labels.csv", "chair\ntable\nsofa\n")
    calls = []

    def encoder(prompts):
        calls.append(list(prompts))
        return np.eye(len(prompts), 4, dtype=np.float32) * 3.0

    cache = build_label_embedding_cache(
        path, tmp_path / "cache.bin", "ViT-Test", encoder
    )

    assert calls == [["chair", "table", "sofa"]]
    assert cache.dimension == 4
    assert len(cache.vocabulary) == 3
    assert cache.embeddings.shape == (3, 4)
    # Every row is finite and unit-norm.
    assert np.isfinite(cache.embeddings).all()
    np.testing.assert_allclose(np.linalg.norm(cache.embeddings, axis=1), 1.0, atol=1e-6)


def test_label_cache_is_reused_without_re_encoding(tmp_path):
    path = _write(tmp_path, "labels.csv", "chair\ntable\n")
    calls = []

    def encoder(prompts):
        calls.append(list(prompts))
        return np.eye(len(prompts), 4, dtype=np.float32) * 2.0

    first = build_label_embedding_cache(path, tmp_path / "c.bin", "M", encoder)
    second = build_label_embedding_cache(path, tmp_path / "c.bin", "M", encoder)

    assert len(calls) == 1  # the second build reads the cache file
    assert second.reused_cache
    np.testing.assert_allclose(first.embeddings, second.embeddings, atol=1e-6)


def test_label_cache_lookup_by_index_and_name(tmp_path):
    path = _write(tmp_path, "labels.csv", "chair\ntable\n")
    cache = build_label_embedding_cache(
        path,
        tmp_path / "c.bin",
        "M",
        lambda prompts: np.eye(len(prompts), 3, dtype=np.float32),
    )

    np.testing.assert_allclose(cache.by_index(0), [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(cache.by_name("Table"), [0.0, 1.0, 0.0], atol=1e-6)
    # A miss returns None so the caller can reject the detection.
    assert cache.by_index(99) is None
    assert cache.by_index(None) is None
    assert cache.by_name("spaceship") is None
    assert cache.by_name("") is None


def test_label_cache_rejects_zero_norm_rows(tmp_path):
    path = _write(tmp_path, "labels.csv", "chair\ntable\n")
    with pytest.raises(Exception):
        build_label_embedding_cache(
            path,
            tmp_path / "c.bin",
            "M",
            lambda prompts: np.zeros((len(prompts), 3), dtype=np.float32),
        )


def test_label_cache_summary_reports_size_and_duration(tmp_path):
    path = _write(tmp_path, "labels.csv", "chair\n")
    cache = build_label_embedding_cache(
        path, tmp_path / "c.bin", "M",
        lambda prompts: np.ones((len(prompts), 8), dtype=np.float32),
    )
    summary = cache.summary()
    assert "1 class labels" in summary
    assert "8-D" in summary
    assert "KiB" in summary
    assert cache.nbytes == 8 * 4


def test_label_cache_classify_returns_the_closest_label(tmp_path):
    path = _write(tmp_path, "labels.csv", "chair\ntable\n")
    cache = build_label_embedding_cache(
        path,
        tmp_path / "c.bin",
        "M",
        lambda prompts: np.eye(len(prompts), 2, dtype=np.float32),
    )

    index, similarity = cache.classify(np.array([0.0, 1.0], dtype=np.float32))
    assert index == 1
    assert similarity == pytest.approx(1.0, abs=1e-6)
    # A dimension mismatch cannot be classified.
    assert cache.classify(np.array([1.0, 0.0, 0.0], dtype=np.float32)) is None
