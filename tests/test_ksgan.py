"""Kiểm thử KSGAN theo hợp đồng augmentation_core của CD3 (dữ liệu tổng hợp, chạy trên CPU)."""
import numpy as np
import pandas as pd
import pytest

from augmentation_core import CDSmote, augmentable_model_columns, materialize_synthetic_rows
from augmentation_core.contracts import generation_plan
from augmentation_core.ksgan import KSGAN, editable_matrix


def make_xy(seed=0):
    rng = np.random.default_rng(seed)
    y = np.array(["I/D"] * 5500 + ["G"] * 400 + ["E"] * 100)
    centre = {"I/D": 0.0, "G": 2.0, "E": 4.0}
    x = np.stack([rng.normal(centre[label], 1.0, size=8) for label in y]).astype(np.float32)
    return x, y


class AddConstant:
    name = "ADD_CONSTANT"

    def __init__(self, value):
        self.value = value

    def fit(self, real, cluster, *, seed):
        return self

    def transform(self, seeds, cluster, *, seed):
        return seeds + self.value

    def audit(self):
        return {"refiner": self.name}


def test_identity_refiner_equals_cdsmote_v2():
    x, y = make_xy()
    ref = CDSmote(n_clusters=5, random_state=42).fit_resample(x, y)
    out = KSGAN(n_clusters=5, random_state=42).fit_resample(x, y)
    for field in ("values", "labels", "parent_a", "parent_b", "alpha"):
        assert np.array_equal(getattr(ref, field), getattr(out, field))
    assert out.method == "KSGAN_FLAT"


def test_counts_follow_ir10_k10_and_majority_untouched():
    x, y = make_xy()
    out = KSGAN(random_state=42).fit_resample(x, y)
    _, _, to_generate = generation_plan(y)
    labels, counts = np.unique(out.labels, return_counts=True)
    assert dict(zip(labels.tolist(), counts.tolist())) == {"G": to_generate["G"], "E": to_generate["E"]}
    assert to_generate["I/D"] == 0


def test_parents_share_class_and_cluster():
    x, y = make_xy()
    model = KSGAN(random_state=42)
    out = model.fit_resample(x, y)
    assert (y[out.parent_a] == out.labels).all() and (y[out.parent_b] == out.labels).all()
    assert (out.parent_a != out.parent_b).all()
    for label in ("G", "E"):
        members = np.flatnonzero(y == label)
        cluster_of = np.full(len(y), -1)
        cluster_of[members] = model._real_clusters(x, members)
        rows = out.labels == label
        assert (cluster_of[out.parent_a[rows]] == cluster_of[out.parent_b[rows]]).all()


def test_shift_is_bounded_and_frozen_outside_editable():
    x, y = make_xy()
    editable = np.ones_like(x, dtype=bool)
    editable[:, 0] = False                      # cột 0 không quan sát được ở mọi dòng
    seeds = CDSmote(random_state=7).fit_resample(x, y)
    out = KSGAN(random_state=7, refiner_factory=lambda: AddConstant(5.0), max_shift=0.25).fit_resample(
        x, y, editable=editable)
    shift = out.values - seeds.values
    assert np.allclose(shift[:, 0], 0.0)
    assert np.allclose(shift[:, 1:], 0.25)


def test_invalid_refiner_output_raises():
    x, y = make_xy()
    with pytest.raises(ValueError):
        KSGAN(refiner_factory=lambda: AddConstant(np.nan)).fit_resample(x, y)


def test_same_seed_same_output_different_seed_differs():
    x, y = make_xy()
    a = KSGAN(random_state=1, refiner_factory=lambda: AddConstant(0.1)).fit_resample(x, y)
    b = KSGAN(random_state=1, refiner_factory=lambda: AddConstant(0.1)).fit_resample(x, y)
    c = KSGAN(random_state=2, refiner_factory=lambda: AddConstant(0.1)).fit_resample(x, y)
    assert np.array_equal(a.values, b.values)
    assert not np.array_equal(a.values, c.values)


def make_train_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(["I/D", "G", "E"], size=n, p=[0.9, 0.07, 0.03])
    frame = pd.DataFrame({"enrollment_id": [f"e{i}" for i in range(n)], "LO_performance_label_3": y,
                          "gender": rng.integers(0, 3, size=n), "user_id_code": rng.integers(0, 500, size=n),
                          "context__offering_id": rng.choice(["o1", "o2"], size=n),
                          "age_at_enroll_scaled": rng.normal(size=n).astype(np.float32)})
    shift = pd.Series(y).map({"I/D": 0.0, "G": 1.5, "E": 3.0}).to_numpy()
    for p in range(1, 5):
        frame[f"watch_count_P{p}"] = (rng.normal(size=n) + shift).astype(np.float32)
        observed = rng.random(n) > 0.3
        frame[f"problem_observed_mask_P{p}"] = observed.astype("int8")
        frame[f"problem_score_P{p}"] = np.where(observed, rng.normal(size=n) + shift, 0.0).astype(np.float32)
        frame[f"phase_available_P{p}"] = np.ones(n, dtype="int8")
        frame[f"missing__watch_count_P{p}"] = (rng.random(n) < 0.1).astype("int8")
    return frame


def structural_cells(frame, columns):
    """Ô vắng cấu trúc: modality problem không có quan sát (giống _structural_mask của imputation_core)."""
    result = pd.DataFrame(False, index=frame.index, columns=list(columns))
    for column in columns:
        if column.startswith("problem_score_P"):
            result[column] = frame[f"problem_observed_mask_P{column[-1]}"].eq(0)
    return result


def test_materialize_with_cd3_ledger_keeps_masks_and_identities():
    train = make_train_frame()
    label = "LO_performance_label_3"
    numeric = augmentable_model_columns(train, label_column=label, phase_id="P4", categorical_columns=("gender",))
    assert "user_id_code" not in numeric and "gender" not in numeric
    assert not any(c.startswith(("missing__", "phase_available_", "problem_observed_mask_")) for c in numeric)
    x = train[numeric].to_numpy(np.float32)
    editable = editable_matrix(train, numeric, structural_cells(train, numeric))
    seeds = CDSmote(random_state=42).fit_resample(x, train[label].to_numpy())
    batch = KSGAN(random_state=42, refiner_factory=lambda: AddConstant(1.0), max_shift=0.3).fit_resample(
        x, train[label].to_numpy(), editable=editable)
    available = [c for c in train if c.startswith(("phase_available_", "problem_observed_mask_"))]
    missing = [c for c in train if c.startswith("missing__")]
    synthetic, ledger = materialize_synthetic_rows(
        train, batch, numeric_columns=numeric, label_column=label, available_mask_columns=available,
        missing_mask_columns=missing, context={"task": "LO", "window_id": "W1", "phase_id": "P4",
                                               "balance_pipeline": "V17", "seed": "42"})
    assert len(ledger) == len(batch.labels) and ledger["synthetic_id"].is_unique
    assert not set(synthetic["enrollment_id"]).intersection(train["enrollment_id"])
    a, b = batch.parent_a, batch.parent_b
    for p in range(1, 5):
        mask = f"problem_observed_mask_P{p}"
        expected = (train[mask].to_numpy()[a] & train[mask].to_numpy()[b]).astype("int8")
        assert np.array_equal(synthetic[mask].to_numpy(), expected)
        flag = f"missing__watch_count_P{p}"
        assert np.array_equal(synthetic[flag].to_numpy(), (train[flag].to_numpy()[a] | train[flag].to_numpy()[b]))
        # Ô vắng cấu trúc ở một trong hai mẫu cha không bị refiner tạo ra "hoạt động" mới.
        j = numeric.index(f"problem_score_P{p}")
        absent = expected == 0
        assert np.allclose(batch.values[absent, j], seeds.values[absent, j])
    assert np.array_equal(synthetic["gender"].to_numpy(), train["gender"].to_numpy()[a])
    assert np.array_equal(synthetic["user_id_code"].to_numpy(), train["user_id_code"].to_numpy()[a])


def test_residual_gan_refiner_is_deterministic_and_bounded():
    torch = pytest.importorskip("torch")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    from augmentation_core.ksgan_torch import ResidualGANRefiner
    x, y = make_xy()
    seeds = CDSmote(random_state=3).fit_resample(x, y)
    run = lambda: KSGAN(random_state=3, max_shift=0.5,
                        refiner_factory=lambda: ResidualGANRefiner(epochs=3, batch_size=128, max_shift=0.5,
                                                                   device="cpu"))
    m1, m2 = run(), run()
    a, b = m1.fit_resample(x, y), m2.fit_resample(x, y)
    assert np.array_equal(a.values, b.values)
    shift = np.abs(a.values - seeds.values)
    assert shift.max() <= 0.5 + 1e-6 and shift.mean() > 0
    history = m1.audit["per_class"]["E"]["refiner_audit"]["history"]
    assert len(history) == 3 and all(np.isfinite([h["d_loss"], h["g_loss"]]).all() for h in history)
