import pytest

from opharm.stats import lock


def test_heldout_is_sealed_without_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(lock, "LOCK", tmp_path / "prereg.lock")
    rows = [{"split": "dev"}, {"split": "heldout"}]
    assert lock.dev_only(rows) == [{"split": "dev"}]
    assert lock.analysis_rows([{"split": "dev"}]) == [{"split": "dev"}]
    with pytest.raises(PermissionError):
        lock.analysis_rows(rows)
    (tmp_path / "prereg.lock").write_text("stale-hash")
    with pytest.raises(PermissionError):
        lock.analysis_rows(rows)
    (tmp_path / "prereg.lock").write_text(lock.prereg_hash())
    assert lock.analysis_rows(rows) == rows


def test_heldout_is_sealed_per_model(monkeypatch, tmp_path):
    monkeypatch.setattr(lock, "LOCK", tmp_path / "prereg.lock")
    (tmp_path / "prereg.lock").write_text(lock.prereg_hash())
    rows = [{"split": "dev"}, {"split": "heldout"}]
    assert lock.analysis_rows(rows, "qwen35_4b") == rows
    with pytest.raises(PermissionError):
        lock.analysis_rows(rows, "model_without_locked_values")
    assert lock.analysis_rows([{"split": "dev"}], "model_without_locked_values") == [{"split": "dev"}]
