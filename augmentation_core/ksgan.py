"""KSGAN balancer theo hợp đồng augmentation_core của CD3 (chỉ TRAIN, vector P1-P4 đầy đủ).

Mẫu mồi lấy đúng từ CDSmote đã đăng ký (K-Means trong từng lớp thiểu số, nội suy
giữa hai thành viên của cùng một cụm). Bộ tinh chỉnh (refiner) sau đó dịch từng
mẫu mồi theo ba quy tắc:
  1. refiner chỉ được fit trên dòng thật của TRAIN thuộc cùng lớp;
  2. không đổi vị trí nào không quan sát được ở cả hai mẫu cha
     (editable theo INTERSECTION_OF_PARENTS, giống quy tắc mặt nạ);
  3. độ dịch mỗi đặc trưng bị chặn bởi max_shift (đơn vị sau RobustScaler).
Với IdentityRefiner, đầu ra trùng từng bit với CDSmote (pipeline V2), nên V2 là
ablation "bỏ bước tinh chỉnh" chính xác. parent_a, parent_b, alpha trong
SyntheticBatch là của mẫu mồi; độ dịch của refiner ghi trong self.audit.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .cdsmote import CDSmote
from .contracts import (DEFAULT_IR_TARGET, DEFAULT_MAX_EXPANSION_PER_CLASS,
                        SyntheticBatch, validate_train_matrix)


class Refiner(Protocol):
    name: str

    def fit(self, real: np.ndarray, cluster: np.ndarray, *, seed: int) -> "Refiner": ...

    def transform(self, seeds: np.ndarray, cluster: np.ndarray, *, seed: int) -> np.ndarray: ...

    def audit(self) -> dict: ...


class IdentityRefiner:
    """Không tinh chỉnh: KSGAN trùng CDSMOTE (ablation AB2)."""

    name = "IDENTITY"

    def fit(self, real, cluster, *, seed):
        return self

    def transform(self, seeds, cluster, *, seed):
        return np.array(seeds, copy=True)

    def audit(self):
        return {"refiner": self.name}


class KSGAN:
    """Mẫu mồi CDSMOTE + refiner có chặn; trả về SyntheticBatch kèm nguồn gốc."""

    method = "KSGAN_FLAT"

    def __init__(self, n_clusters: int = 5, random_state: int = 20260922,
                 ir_target: int = DEFAULT_IR_TARGET,
                 max_expansion_per_class: int = DEFAULT_MAX_EXPANSION_PER_CLASS,
                 refiner_factory: Callable[[], Refiner] = IdentityRefiner,
                 max_shift: float = 0.5, method: str | None = None):
        if max_shift <= 0:
            raise ValueError("max_shift must be positive")
        self.seeder = CDSmote(n_clusters=n_clusters, random_state=random_state, ir_target=ir_target,
                              max_expansion_per_class=max_expansion_per_class)
        self.refiner_factory = refiner_factory
        self.max_shift = float(max_shift)
        if method:
            self.method = method
        self.plan: dict[str, object] = {}
        self.audit: dict[str, object] = {}

    # augmentation_app.py đọc các thuộc tính này khi ghi algorithm_config.
    n_clusters = property(lambda self: self.seeder.n_clusters)
    random_state = property(lambda self: self.seeder.random_state)
    ir_target = property(lambda self: self.seeder.ir_target)
    max_expansion_per_class = property(lambda self: self.seeder.max_expansion_per_class)
    sampling_strategy_id = property(lambda self: self.seeder.sampling_strategy_id)

    def _real_clusters(self, values: np.ndarray, members: np.ndarray) -> np.ndarray:
        """Tính lại phân cụm K-Means của CDSmote cho một lớp thiểu số (cùng tham số, cùng seed)."""
        k = min(self.n_clusters, max(1, len(members) // 2))
        return KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit_predict(values[members])

    def fit_resample(self, x, y, editable=None) -> SyntheticBatch:
        values, labels = validate_train_matrix(x, y)
        if editable is not None:
            editable = np.asarray(editable, dtype=bool)
            if editable.shape != values.shape:
                raise ValueError("editable must have the same shape as x")
        seeds = self.seeder.fit_resample(values, labels)
        self.plan = dict(self.seeder.plan)
        refined = seeds.values.astype(np.float32, copy=True)
        per_class: dict[str, dict] = {}
        for label in np.unique(seeds.labels):
            rows = np.flatnonzero(seeds.labels == label)
            members = np.flatnonzero(labels == label)
            cluster_of = np.full(len(labels), -1, dtype=int)
            cluster_of[members] = self._real_clusters(values, members)
            cluster = cluster_of[seeds.parent_a[rows]]
            refiner = self.refiner_factory().fit(values[members], cluster_of[members], seed=self.random_state)
            base = seeds.values[rows]
            out = np.asarray(refiner.transform(base, cluster, seed=self.random_state), dtype=np.float32)
            if out.shape != base.shape or not np.isfinite(out).all():
                raise ValueError(f"refiner {refiner.name} returned an invalid matrix for class {label!r}")
            raw_shift = out - base
            shift = np.clip(raw_shift, -self.max_shift, self.max_shift)
            if editable is not None:
                free = editable[seeds.parent_a[rows]] & editable[seeds.parent_b[rows]]
                shift = np.where(free, shift, 0.0)
            refined[rows] = base + shift
            norm = np.linalg.norm(shift, axis=1)
            per_class[str(label)] = {
                "real_rows": int(len(members)), "synthetic_rows": int(len(rows)),
                "cluster_sizes": np.bincount(cluster_of[members]).tolist(),
                "shift_l2_mean": float(norm.mean()), "shift_l2_p95": float(np.quantile(norm, 0.95)),
                "shift_clipped_share": float(np.mean(np.abs(raw_shift) > self.max_shift)),
                "refiner_audit": refiner.audit(),
            }
        self.audit = {"algorithm": self.method, "seeder": "CDSMOTE", "n_clusters": self.n_clusters,
                      "max_shift": self.max_shift, "per_class": per_class}
        return SyntheticBatch(values=refined, labels=seeds.labels, parent_a=seeds.parent_a,
                              parent_b=seeds.parent_b, alpha=seeds.alpha, method=self.method)


def editable_matrix(train: pd.DataFrame, numeric_columns: Sequence[str],
                    structural: pd.DataFrame | None = None) -> np.ndarray:
    """True tại ô quan sát thật: không phải ô điền khuyết (missing__c = 1) và không vắng cấu trúc.

    ``structural`` là kết quả của imputation_core.wide_imputation._structural_mask
    (True khi modality của ô không có quan sát); None nghĩa là không xét.
    """
    result = np.ones((len(train), len(numeric_columns)), dtype=bool)
    for j, column in enumerate(numeric_columns):
        flag = f"missing__{column}"
        if flag in train:
            result[:, j] &= train[flag].to_numpy() == 0
        if structural is not None and column in structural:
            result[:, j] &= ~structural[column].to_numpy(dtype=bool)
    return result
