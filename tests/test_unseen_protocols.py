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
