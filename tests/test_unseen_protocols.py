import numpy as np
import pytest

from unseen_exp.generate_unseen_list import generate_unseen_indices, largest_remainder_quotas
from unseen_exp.unseen_exp import (EXPERIMENT_CONFIGS, generate_random_mask, mask_path_for,
                                  validate_experiment)


def test_stratified_generation_reproducible_and_exact():
    labels = np.repeat(np.arange(7), [11, 13, 17, 19, 23, 29, 31])
    for ratio in (50, 80):
        a = generate_unseen_indices(labels, ratio, 22)
        b = generate_unseen_indices(labels, ratio, 22)
        c = generate_unseen_indices(labels, ratio, 42)
        assert len(a) == round(len(labels) * ratio / 100)
        assert np.array_equal(a, b) and not np.array_equal(a, c)
        assert len(np.unique(a)) == len(a) and (a >= 0).all() and (a < len(labels)).all()
        known = np.setdiff1d(np.arange(len(labels)), a)
        assert np.array_equal(np.sort(np.r_[known, a]), np.arange(len(labels)))
        quotas = largest_remainder_quotas(labels, ratio)
        assert {k: int((labels[a] == k).sum()) for k in quotas} == quotas


@pytest.mark.parametrize("args", [(1, "cifar10", "learned_group"), (2, "cifar100", "learned_group"),
                                   (3, "tiny-imagenet", "learned_group"), (1, "cifar100", "naive_group"),
                                   (2, "cifar10", "naive_group")])
def test_invalid_protocol_combinations(args):
    with pytest.raises(ValueError): validate_experiment(*args)


@pytest.mark.parametrize("exp,dataset,mode", [(1,"cifar100","learned_group"),
                                                (2,"cifar10","learned_group"),
                                                (3,"cifar100","naive_group")])
def test_invalid_keep_ratio(exp, dataset, mode):
    with pytest.raises(ValueError): validate_experiment(exp, dataset, mode, "99")


def test_paths_and_exp2_random_counts():
    path = mask_path_for(3, "naive_group", "cifar100", 22, 50)
    assert path.as_posix().endswith("unseen_exp/mask/3/unseen_naive_group/cifar100/22/mask_50.npz")
    unseen = np.arange(10_000, 50_000)
    for kr, expected in ((20,10_000),(30,15_000),(40,20_000),(60,30_000)):
        mask = generate_random_mask(unseen, 50_000, expected, 22, 2, kr)
        assert mask.sum() == expected and not mask[:10_000].any()
        assert np.array_equal(mask, generate_random_mask(unseen, 50_000, expected, 22, 2, kr))


def test_configs_are_exact():
    assert EXPERIMENT_CONFIGS[1].default_keep_ratios == (60,70,80,90)
    assert EXPERIMENT_CONFIGS[2].selection_scope == "unseen"
    assert EXPERIMENT_CONFIGS[3].group_solver == "center_repair"

from pathlib import Path
from types import SimpleNamespace
import json

import torch
from torch.utils.data import Dataset
import unseen_exp.unseen_exp as ue


class FakeDataset(Dataset):
    def __init__(self, values, transform=None):
        self.values = list(values); self.targets = [v % 2 for v in range(len(values))]
        self.classes = ["a", "b"]; self.transform = transform
    def __len__(self): return len(self.values)
    def __getitem__(self, index):
        value = self.values[index]
        return (self.transform(value) if self.transform else value), self.targets[index]


def make_args(exp, mode="learned_group", skip_saved=True):
    config = ue.EXPERIMENT_CONFIGS[exp]
    return SimpleNamespace(exp=exp, config=config, dataset=config.allowed_datasets[0], mode=mode,
        seed=22, data_root="data", clip_model="clip", proxy_model="proxy", device="cpu",
        skip_saved=skip_saved, debug_prompts=False, group_candidate_pool_size=1,
        group_init_count=2, dist_weight_factor=1.0, keep_ratios=config.default_keep_ratios)


def patch_solver_dependencies(monkeypatch, captured):
    class FakeDiv:
        def __init__(self, **kwargs): self.extractor = SimpleNamespace(embed_dim=2, preprocess=lambda x: x)
    monkeypatch.setattr(ue, "Div", FakeDiv)
    monkeypatch.setattr(ue, "resolve_class_names_for_prompts", lambda *a: ["a", "b"])
    monkeypatch.setattr(ue, "load_trained_adapters", lambda *a, **k: (object(), object(), None))
    def solver(**kwargs):
        captured.append(kwargs); n = len(kwargs["labels"]); target = round(n * kwargs["keep_ratio"] / 100)
        mask = np.zeros(n, np.uint8); mask[:target] = 1
        return mask, {}, {}
    return solver


@pytest.mark.parametrize("exp", [1, 2])
def test_standard_group_receives_weight_group(monkeypatch, exp):
    captured = []; solver = patch_solver_dependencies(monkeypatch, captured)
    monkeypatch.setattr(ue.mask_solvers, "select_group_mask", solver)
    dataset = FakeDataset(range(10)); static = {"sa": np.zeros(10), "dds": np.zeros(10), "labels": np.arange(10) % 2}
    ue.run_group_solver(make_args(exp), static, dataset, Path("adapter"), {"sa": .3, "div": .3, "dds": .4}, 5)
    assert captured[0]["weight_group"] == "learned"


def test_center_repair_does_not_receive_weight_group(monkeypatch):
    captured = []; solver = patch_solver_dependencies(monkeypatch, captured)
    monkeypatch.setattr(ue.mask_solvers, "select_group_mask_by_center_repair", solver)
    dataset = FakeDataset(range(10)); static = {"sa": np.zeros(10), "dds": np.zeros(10), "labels": np.arange(10) % 2}
    ue.run_group_solver(make_args(3), static, dataset, Path("adapter"), {"sa": 1/3, "div": 1/3, "dds": 1/3}, 5)
    assert "weight_group" not in captured[0]


def test_exp3_proxy_uses_corrupted_full_factory():
    clean = FakeDataset(range(6), transform=lambda value: value + 1)
    calls = []
    def full_factory(transform):
        calls.append(transform); return FakeDataset([100 + i for i in range(6)], transform=transform)
    known = ue.build_proxy_known_dataset(clean, full_factory, np.array([1, 4]))
    assert calls == [clean.transform] and len(known) == 2
    assert known[0][0] == 102 and known[1][0] == 105
    assert np.array_equal(known.indices, [1, 4])


def valid_weight_entry(args, known, unseen, epochs=10, info=None):
    return dict(sa=.2, div=.3, dds=.5, bias=.1, ratio_lambda=1e-3, exp=args.exp,
        dataset=args.dataset, seed=args.seed, known_ratio=args.config.known_ratio,
        unseen_ratio=args.config.unseen_ratio, known_indices=known.tolist(), unseen_indices=unseen.tolist(),
        proxy_model=args.proxy_model, proxy_epochs=epochs, clip_model=args.clip_model,
        static_reference_scope=args.config.static_reference_scope, selection_scope=args.config.selection_scope,
        **ue._json_corruption_context(info))


def test_valid_weight_cache_skips_proxy_and_dynamic(tmp_path, monkeypatch):
    args = make_args(1); known, unseen = np.arange(3), np.arange(3, 6)
    weight = tmp_path / "weights.json"
    ue._atomic_json(weight, {args.dataset: {str(args.seed): valid_weight_entry(args, known, unseen)}})
    monkeypatch.setattr(ue, "resolve_proxy_epochs", lambda _: 10)
    monkeypatch.setattr(ue, "cache_paths", lambda *a, **k: {"weights": weight})
    monkeypatch.setattr(ue, "train_proxy_on_known", lambda *a, **k: pytest.fail("proxy called"))
    result = ue.prepare_proxy_and_weights(args, known, unseen, {}, None)
    assert result == {"sa": .2, "div": .3, "dds": .5}


@pytest.mark.parametrize("mutation", ["missing", "sum", "indices"])
def test_invalid_weight_cache_recomputes(tmp_path, mutation):
    args = make_args(1); known, unseen = np.arange(3), np.arange(3, 6); entry = valid_weight_entry(args, known, unseen)
    if mutation == "missing": entry.pop("bias")
    elif mutation == "sum": entry["sa"] = .4
    else: entry["known_indices"] = [1, 2, 3]
    path = tmp_path / "weights.json"; ue._atomic_json(path, {args.dataset: {str(args.seed): entry}})
    assert ue.load_scoring_weights(path, args, known, unseen, 10, None) is None


def test_exp3_corruption_change_invalidates_weight_cache(tmp_path):
    args = make_args(3); known, unseen = np.arange(2), np.arange(2, 4)
    info = SimpleNamespace(is_corrupted=np.array([True, False, True, False]),
                           corruption_types=np.array([0, -1, 1, -1], dtype=np.int16))
    changed = SimpleNamespace(is_corrupted=info.is_corrupted.copy(),
                              corruption_types=np.array([1, -1, 0, -1], dtype=np.int16))
    path = tmp_path / "weights.json"
    ue._atomic_json(path, {args.dataset: {str(args.seed): valid_weight_entry(args, known, unseen, info=info)}})
    assert ue.load_scoring_weights(path, args, known, unseen, 10, info) is not None
    assert ue.load_scoring_weights(path, args, known, unseen, 10, changed) is None


def dynamic_value(n):
    return SimpleNamespace(raw_foldwise=np.full((5,n), np.nan), fold_normalized=np.full((5,n), np.nan),
                           aggregated=np.arange(n, dtype=float), final_normalized=np.arange(n, dtype=float))


def write_dynamic(path, name, args, known, unseen, labels, epochs, value):
    ue._atomic_savez(path, labels=labels, raw_foldwise=value.raw_foldwise,
        fold_normalized=value.fold_normalized, aggregated=value.aggregated,
        final_normalized=value.final_normalized, component=name, exp=args.exp, dataset=args.dataset,
        seed=args.seed, known_ratio=args.config.known_ratio, unseen_ratio=args.config.unseen_ratio,
        known_indices=known, unseen_indices=unseen, proxy_model=args.proxy_model, proxy_epochs=epochs,
        clip_model=args.clip_model, static_reference_scope=args.config.static_reference_scope,
        selection_scope=args.config.selection_scope, corruption_types=np.empty(0,np.int16), is_corrupted=np.empty(0,bool))


def test_partial_dynamic_cache_reuse(tmp_path, monkeypatch):
    args = make_args(1); known, unseen = np.arange(3), np.arange(3,6); labels=np.array([0,1,0]); epochs=2
    for name in ("A", "C"): write_dynamic(tmp_path/f"{name}.npz", name, args, known, unseen, labels, epochs, dynamic_value(3))
    monkeypatch.setattr(ue, "cache_paths", lambda *a, **k: {"dynamic": tmp_path})
    monkeypatch.setattr(ue.dynamic_utils, "load_cv_fold_logs", lambda *a: (object(), labels))
    calls=[]
    class Calc:
        def compute(self, **kwargs): calls.append("T"); return dynamic_value(3)
    monkeypatch.setitem(ue.COMPONENTS, "T", Calc)
    result = ue.get_dynamic_components(args, Path("proxy"), epochs, known, unseen, labels, None)
    assert set(result) == {"A","C","T"} and calls == ["T"]


def test_adapter_cache_requires_all_files(tmp_path, monkeypatch):
    args=make_args(1); known=np.arange(2); unseen=np.arange(2,4)
    meta=valid_weight_entry(args, known, unseen); meta.update(num_samples=2, adapter_type="linear", training_objective="InfoNCE")
    (tmp_path/"meta.json").write_text(json.dumps(meta))
    assert not ue.adapter_cache_valid(tmp_path,args,known,unseen,None)
    (tmp_path/"adapter_image.pt").write_text("bad"); (tmp_path/"adapter_context.pt").write_text("bad")
    assert not ue.adapter_cache_valid(tmp_path,args,known,unseen,None)


def proxy_meta(args, known, unseen, epochs):
    return dict(exp=args.exp,dataset=args.dataset,seed=args.seed,num_samples=len(known),epochs=epochs,k_folds=5,
        known_ratio=args.config.known_ratio,unseen_ratio=args.config.unseen_ratio,known_indices=known.tolist(),
        unseen_indices=unseen.tolist(),model=args.proxy_model,proxy_model=args.proxy_model,num_classes=2,
        static_reference_scope=args.config.static_reference_scope,selection_scope=args.config.selection_scope,
        **ue._json_corruption_context(None))


def write_proxy_fold(path, train, val, epochs=2):
    ue._atomic_savez(path,train_indices=np.array(train),val_indices=np.array(val),
        train_logits=np.zeros((epochs,len(train),2)),val_logits=np.zeros((epochs,len(val),2)))


def test_proxy_cache_requires_complete_valid_folds(tmp_path):
    args=make_args(1); known=np.arange(5); unseen=np.arange(5,10); epochs=2
    (tmp_path/"meta.json").write_text(json.dumps(proxy_meta(args,known,unseen,epochs)))
    for fold in range(5):
        val=[fold]; train=[i for i in range(5) if i != fold]
        write_proxy_fold(tmp_path/f"fold_{fold+1}.npz",train,val,epochs)
    assert ue.proxy_cache_valid(tmp_path,args,known,unseen,epochs,None)
    (tmp_path/"fold_5.npz").unlink(); assert not ue.proxy_cache_valid(tmp_path,args,known,unseen,epochs,None)
    write_proxy_fold(tmp_path/"fold_5.npz",[0,1,2,3],[4],1)
    assert not ue.proxy_cache_valid(tmp_path,args,known,unseen,epochs,None)


def test_static_cache_missing_field_recomputes(tmp_path):
    args=make_args(2); known=np.arange(2); unseen=np.arange(2,6); labels=np.array([0,1]); samples=known
    ue._atomic_savez(tmp_path/"cache.npz",sa=np.zeros(2),div=np.zeros(2),labels=labels)
    assert ue.load_static_cache(tmp_path/"cache.npz",args,"calibration",labels,samples,known,unseen,None) is None


def test_exp2_static_scopes():
    args=make_args(2); known=np.array([0,2]); unseen=np.array([1,3,4])
    assert ue.static_reference_scope(args,"calibration") == "known"
    assert ue.static_reference_scope(args,"selection") == "unseen"
    assert np.array_equal(ue.static_sample_indices(ue.IndexedSubsetDataset(FakeDataset(range(5)),known)),known)
    assert np.array_equal(ue.static_sample_indices(ue.IndexedSubsetDataset(FakeDataset(range(5)),unseen)),unseen)
