from __future__ import annotations

import numpy as np
import pytest

from respiralab.models import ood_gate


def _clustered_embeddings(n=200, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n, dim)).astype(np.float32)


def test_mahalanobis_distances_reduces_to_euclidean_under_identity_covariance():
    mean = np.zeros(3)
    cov_inv = np.eye(3)
    points = np.array([[1.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    distances = ood_gate.mahalanobis_distances(points, mean, cov_inv)
    assert distances == pytest.approx([1.0, 5.0])


def test_fit_gate_accepts_train_distribution_and_rejects_far_outlier():
    embeddings = _clustered_embeddings()
    gate = ood_gate.fit_gate(embeddings)

    in_dist, distance = ood_gate.is_in_distribution(embeddings[0], gate)
    assert in_dist
    assert distance <= gate["threshold"]

    outlier = np.full(embeddings.shape[1], 1000.0, dtype=np.float32)
    rejected, far_distance = ood_gate.is_in_distribution(outlier, gate)
    assert not rejected
    assert far_distance > gate["threshold"]


def test_fit_gate_threshold_matches_requested_percentile_within_train_set():
    embeddings = _clustered_embeddings(n=500)
    gate = ood_gate.fit_gate(embeddings, percentile=90.0)
    distances = ood_gate.mahalanobis_distances(embeddings, gate["mean"], gate["cov_inv"])
    frac_within = (distances <= gate["threshold"]).mean()
    assert frac_within == pytest.approx(0.90, abs=0.03)


def test_save_and_load_gate_round_trips(tmp_path):
    embeddings = _clustered_embeddings()
    gate = ood_gate.fit_gate(embeddings)
    metadata = ood_gate.save_gate(gate, tmp_path)

    assert metadata["threshold"] == gate["threshold"]
    assert (tmp_path / "ood_gate_mean.npy").exists()
    assert (tmp_path / "ood_gate_cov_inv.npy").exists()

    loaded = ood_gate.load_gate(metadata)
    np.testing.assert_allclose(loaded["mean"], gate["mean"])
    np.testing.assert_allclose(loaded["cov_inv"], gate["cov_inv"])
    assert loaded["threshold"] == gate["threshold"]
