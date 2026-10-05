import numpy as np

from opharm.stats.bootstrap import cluster_ci, cluster_draws, holm, p_beyond, template_ci


def test_holm_step_down():
    adj, rej = holm([0.01, 0.04, 0.03, 0.005, 0.2])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.025, 0.2])
    assert rej.tolist() == [True, False, False, True, False]


def test_p_beyond_counts_null_side_with_correction():
    draws = np.array([-1.0, 0.0, 1.0, 2.0])
    assert p_beyond(draws, 0.0, "greater") == 3 / 5
    assert p_beyond(draws, 1.0, "less") == 3 / 5


def test_cluster_draws_stay_within_strata():
    strata = np.array(["a", "a", "b", "b", "b"])
    d = cluster_draws(strata, n=200)
    assert d.shape == (200, 5)
    assert set(strata[d[:, :2]].ravel()) == {"a"} and set(strata[d[:, 2:]].ravel()) == {"b"}


def test_template_ci_estimate_is_plain_mean():
    values = [0.0, 1.0, 1.0, 0.5, 0.25, 1.0, 0.0]
    est, lo, hi = template_ci(values, ["a", "a", "b", "b", "b", "c", "d"], n=500)
    assert np.isclose(est, np.mean(values)) and lo <= est <= hi


def test_template_ci_is_deterministic():
    args = ([0.0, 1.0, 1.0, 0.0, 1.0, 1.0], ["a", "a", "b", "c", "c", "d"])
    assert template_ci(*args, n=500) == template_ci(*args, n=500)


def test_template_ci_one_value_per_template_matches_cluster_ci():
    values = [0.0, 1.0, 0.5, 1.0, 0.0, 0.25]
    assert np.allclose(template_ci(values, ["t0", "t1", "t2", "t3", "t4", "t5"], n=500), cluster_ci(values, ["all"] * 6, n=500))


def test_template_ci_wider_than_skeleton_ci_for_clustered_values():
    values = np.repeat([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], 4)
    templates = np.repeat(["a", "b", "c", "d", "e", "f"], 4).tolist()
    _, lo, hi = template_ci(values, templates)
    _, slo, shi = cluster_ci(values, ["all"] * 24)
    assert hi - lo > shi - slo


def test_template_ci_single_template_is_a_point():
    est, lo, hi = template_ci([0.0, 1.0, 1.0], ["a", "a", "a"], n=500)
    assert est == lo == hi
