"""regression tests: point estimates should stay where Hernan & Robins (What If) put them"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prep    # noqa: E402
import models  # noqa: E402


@pytest.fixture(scope="module")
def d():
    return prep.load(download_if_missing=False)


def test_data_shape(d):
    assert len(d) == 1629
    assert d.qsmk.sum() == 428
    assert d.cens.sum() == 63


def test_ipw_matches_book(d):
    # What If, ch. 12: IP weighted effect ~3.4-3.5 kg
    assert models.est_ipw(d) == pytest.approx(3.5, abs=0.1)


def test_stabilized_weights_mean_one(d):
    w = models.ip_weights(d, censoring=False)
    assert w.mean() == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("name", ["Outcome regression", "IP weighting", "G-formula", "AIPW (doubly robust)"])
def test_adjusted_estimators_agree(d, name):
    assert models.ESTIMATORS[name](d) == pytest.approx(3.5, abs=0.15)


def test_matching_reports_unmatched(d):
    est, info = models.psmatch(d)
    assert info["n_treated"] == 403
    assert 0 <= info["n_unmatched"] < 20
    assert np.isfinite(est)


def test_evalue_basic():
    e, e_ci = models.evalue(3.5, 2.5, 4.5, 7.9)
    assert e > e_ci > 1
    assert models.evalue(0.5, -0.5, 1.5, 7.9)[1] == 1.0   # CI crosses 0


def test_bootstrap_runs(d):
    b = models.bootstrap(d, n_boot=3, seed=1)
    assert b.shape == (3, len(models.ESTIMATORS))
