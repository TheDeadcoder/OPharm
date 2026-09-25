import numpy as np

from opharm.stats.bootstrap import cluster_draws, holm, p_beyond


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
