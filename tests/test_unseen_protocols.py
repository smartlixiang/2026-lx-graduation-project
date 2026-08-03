"""Protocol and cache regression tests for the staged unseen experiment."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import unseen_exp.unseen_exp as ue


def make_args(exp=1, mode="learned_group", stages=ue.ALL_STAGES):
    config = ue.EXPERIMENT_CONFIGS[exp]
    return SimpleNamespace(exp=exp, config=config, dataset=config.allowed_datasets[0], mode=mode,
        seed=22, data_root="data", clip_model="clip", proxy_model="proxy", device="cpu",
        skip_saved_stages=frozenset(stages), debug_prompts=False, group_candidate_pool_size=1,
        group_init_count=2, dist_weight_factor=1.0, keep_ratios=config.default_keep_ratios)


def weight_entry(args, known, unseen, *, legacy=False, info=None):
    entry = dict(sa=.2, div=.3, dds=.5, bias=.1, ratio_lambda=1e-3, exp=args.exp,
        dataset=args.dataset, seed=args.seed, known_ratio=args.config.known_ratio,
        unseen_ratio=args.config.unseen_ratio, unseen_indices=unseen.tolist(),
        proxy_model=args.proxy_model, proxy_epochs=10, clip_model=args.clip_model,
        static_reference_scope=args.config.static_reference_scope,
        selection_scope=args.config.selection_scope, **ue._json_corruption_context(info))
    if legacy:
        entry["known_indices"] = known.tolist()
    else:
        entry.update(known_count=len(known), unseen_count=len(unseen),
                     split_fingerprint=ue.split_fingerprint(known, unseen))
    return entry


def test_only_experiments_one_and_three_remain():
    assert set(ue.EXPERIMENT_CONFIGS) == {1, 3}
    assert ue.EXPERIMENT_CONFIGS[1].default_keep_ratios == (60, 70, 80, 90)
    assert ue.EXPERIMENT_CONFIGS[3].default_keep_ratios == (30, 40, 50, 60)
    with pytest.raises(ValueError):
        ue.validate_experiment(2, "cifar10", "learned_group")
    ue.validate_experiment(1, "cifar100", "learned_group")
    ue.validate_experiment(3, "cifar100", "naive_group")


@pytest.mark.parametrize("text, expected", [
    (None, frozenset()), ("1,2,3,4,5,6", ue.ALL_STAGES),
    ("1,2,3", frozenset({1, 2, 3})), ("3,1,3", frozenset({1, 3})),
])
def test_parse_skip_saved_stages(text, expected):
    assert ue.parse_skip_saved_stages(text) == expected


@pytest.mark.parametrize("text", ["0", "7", "1,a,3", ",", "1,,3"])
def test_parse_skip_saved_rejects_invalid_values(text):
    with pytest.raises(ValueError, match="skip-saved"):
        ue.parse_skip_saved_stages(text)


def test_stage_reuse_honors_request_and_upstream_dirty():
    args = make_args(stages={2, 5})
    assert ue.stage_reuse_requested(args, 2)
    assert not ue.stage_reuse_requested(args, 1)
    assert not ue.stage_reuse_requested(args, 2, upstream_dirty=True)
    assert not hasattr(args, "skip_saved")


def test_split_fingerprint_is_deterministic_and_sensitive():
    known, unseen = np.array([0, 2]), np.array([1, 3])
    value = ue.split_fingerprint(known, unseen)
    assert value == ue.split_fingerprint(known.copy(), unseen.copy())
    assert value != ue.split_fingerprint(np.array([0, 3]), np.array([1, 2]))
    assert value != ue.split_fingerprint(known, np.array([1, 4]))


def test_new_weight_cache_has_compact_split_metadata(tmp_path):
    args = make_args(); known, unseen = np.arange(3), np.arange(3, 6)
    path = tmp_path / "weights.json"
    ue.write_weight_payload(path, {args.dataset: {"22": weight_entry(args, known, unseen)}})
    stored = json.loads(path.read_text())[args.dataset]["22"]
    assert "known_indices" not in stored
    assert (stored["known_count"], stored["unseen_count"]) == (3, 3)
    assert stored["split_fingerprint"] == ue.split_fingerprint(known, unseen)
    assert ue.load_scoring_weights(path, args, known, unseen, 10, None) == {"sa": .2, "div": .3, "dds": .5}
    assert ue.load_scoring_weights(path, args, np.array([0, 1, 3]), np.array([2, 4, 5]), 10, None) is None


def test_legacy_weight_cache_migrates_every_entry(tmp_path):
    args = make_args(); known, unseen = np.arange(2), np.arange(2, 4)
    other = weight_entry(args, known, unseen, legacy=True)
    path = tmp_path / "weights.json"
    ue._atomic_json(path, {args.dataset: {"22": weight_entry(args, known, unseen, legacy=True), "99": other}})
    assert ue.load_scoring_weights(path, args, known, unseen, 10, None)
    payload = json.loads(path.read_text())
    assert "known_indices" not in json.dumps(payload)
    assert payload[args.dataset]["22"]["split_fingerprint"] == ue.split_fingerprint(known, unseen)


def test_corruption_change_invalidates_weights(tmp_path):
    args = make_args(3); known, unseen = np.arange(2), np.arange(2, 4)
    info = SimpleNamespace(is_corrupted=np.array([True, False, True, False]),
        corruption_types=np.array([0, -1, 1, -1], dtype=np.int16))
    changed = SimpleNamespace(is_corrupted=info.is_corrupted.copy(),
        corruption_types=np.array([1, -1, 0, -1], dtype=np.int16))
    path = tmp_path / "weights.json"
    ue._atomic_json(path, {args.dataset: {"22": weight_entry(args, known, unseen, info=info)}})
    assert ue.load_scoring_weights(path, args, known, unseen, 10, info)
    assert ue.load_scoring_weights(path, args, known, unseen, 10, changed) is None


def test_random_mask_uses_full_candidate_pool():
    mask = ue.generate_random_mask(np.arange(100), 100, 60, 22, 1, 60)
    assert mask.shape == (100,) and mask.sum() == 60
    assert np.array_equal(mask, ue.generate_random_mask(np.arange(100), 100, 60, 22, 1, 60))


def test_cache_paths_have_one_static_cache_and_pseudo_label_location():
    paths = ue.cache_paths(1, "cifar100", 22, "resnet18", 10)
    assert "calibration_static" not in paths
    assert paths["dynamic"].joinpath("pseudo_labels.npz").as_posix().endswith(
        "dynamic_cache/1/cifar100/resnet18/22/10/pseudo_labels.npz")
