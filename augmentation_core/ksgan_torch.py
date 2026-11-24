"""Refiner GAN dạng dư (residual) cho KSGAN; chạy trên CPU hoặc GPU Kaggle.

G(seed, z, cụm) -> delta = max_shift * tanh(.); mẫu tinh chỉnh = seed + delta.
D(x, cụm) phân biệt dòng thật của lớp với mẫu đã tinh chỉnh.
Loss của G = non-saturating GAN + lambda_prox * mean(delta^2): giữ gần mẫu mồi (O13).
Một GAN cho mỗi lớp, nhận nhãn cụm làm điều kiện (AB4), có nhiễu z (AB3).
Mẫu mồi khi huấn luyện được tạo theo đúng quy tắc của CDSMOTE: nội suy hai dòng thật
cùng cụm. Mọi số ngẫu nhiên lấy từ seed truyền vào; không bắt lỗi im lặng.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _Generator(nn.Module):
    def __init__(self, n_features, n_clusters, noise_dim, hidden, max_shift):
        super().__init__()
        self.embed = nn.Embedding(n_clusters, 8)
        self.net = nn.Sequential(nn.Linear(n_features + noise_dim + 8, hidden), nn.LeakyReLU(0.2),
                                 nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, n_features))
        self.max_shift = max_shift

    def forward(self, seed, z, cluster):
        delta = self.max_shift * torch.tanh(self.net(torch.cat([seed, z, self.embed(cluster)], dim=1)))
        return seed + delta, delta


class _Discriminator(nn.Module):
    def __init__(self, n_features, n_clusters, hidden):
        super().__init__()
        self.embed = nn.Embedding(n_clusters, 8)
        self.net = nn.Sequential(nn.Linear(n_features + 8, hidden), nn.LeakyReLU(0.2),
                                 nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.2), nn.Linear(hidden // 2, 1))

    def forward(self, x, cluster):
        return self.net(torch.cat([x, self.embed(cluster)], dim=1)).squeeze(1)


def _same_cluster_seeds(real, cluster, n, rng):
    """Mẫu mồi huấn luyện: nội suy hai dòng thật khác nhau trong cùng cụm."""
    groups = [np.flatnonzero(cluster == c) for c in np.unique(cluster)]
    groups = [g for g in groups if len(g) >= 2]
    if not groups:
        raise ValueError("refiner needs at least one cluster with two real rows")
    sizes = np.array([len(g) for g in groups], dtype=float)
    pick = rng.choice(len(groups), size=n, p=sizes / sizes.sum())
    left = np.empty(n, dtype=int)
    right = np.empty(n, dtype=int)
    for gi, g in enumerate(groups):
        at = np.flatnonzero(pick == gi)
        if len(at) == 0:
            continue
        lp = rng.integers(0, len(g), size=len(at))
        rp = (lp + rng.integers(1, len(g), size=len(at))) % len(g)
        left[at], right[at] = g[lp], g[rp]
    alpha = rng.random((n, 1)).astype(np.float32)
    return real[left] + alpha * (real[right] - real[left]), cluster[left]


class ResidualGANRefiner:
    name = "RESIDUAL_GAN"

    def __init__(self, epochs: int = 100, batch_size: int = 512, lr: float = 1e-4, noise_dim: int = 16,
                 hidden: int = 256, max_shift: float = 0.5, lambda_prox: float = 1.0, device: str | None = None):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.noise_dim, self.hidden, self.max_shift, self.lambda_prox = noise_dim, hidden, max_shift, lambda_prox
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.history: list[dict] = []

    def fit(self, real, cluster, *, seed):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        real = np.asarray(real, dtype=np.float32)
        cluster = np.asarray(cluster, dtype=np.int64)
        n_clusters = int(cluster.max()) + 1
        self.G = _Generator(real.shape[1], n_clusters, self.noise_dim, self.hidden, self.max_shift).to(self.device)
        self.D = _Discriminator(real.shape[1], n_clusters, self.hidden).to(self.device)
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        x_all = torch.from_numpy(real).to(self.device)
        c_all = torch.from_numpy(cluster).to(self.device)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(len(real))
            d_sum = g_sum = prox_sum = 0.0
            batches = 0
            for start in range(0, len(order), self.batch_size):
                idx = torch.from_numpy(order[start:start + self.batch_size]).to(self.device)
                x_real, c_real = x_all[idx], c_all[idx]
                s_np, sc_np = _same_cluster_seeds(real, cluster, len(idx), rng)
                s = torch.from_numpy(s_np).to(self.device)
                sc = torch.from_numpy(sc_np).to(self.device)
                z = torch.randn(len(idx), self.noise_dim, generator=gen, device=self.device)
                fake, _ = self.G(s, z, sc)
                d_loss = bce(self.D(x_real, c_real), torch.ones(len(idx), device=self.device)) + \
                    bce(self.D(fake.detach(), sc), torch.zeros(len(idx), device=self.device))
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
                fake, delta = self.G(s, z, sc)
                prox = delta.pow(2).mean()
                g_loss = bce(self.D(fake, sc), torch.ones(len(idx), device=self.device)) + self.lambda_prox * prox
                opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                opt_g.step()
                d_sum += d_loss.item()
                g_sum += g_loss.item()
                prox_sum += prox.item()
                batches += 1
            self.history.append({"epoch": epoch, "d_loss": d_sum / batches, "g_loss": g_sum / batches,
                                 "prox": prox_sum / batches})
        return self

    @torch.no_grad()
    def transform(self, seeds, cluster, *, seed):
        self.G.eval()
        gen = torch.Generator(device=self.device).manual_seed(seed + 1)
        s = torch.from_numpy(np.asarray(seeds, dtype=np.float32)).to(self.device)
        c = torch.from_numpy(np.asarray(cluster, dtype=np.int64)).to(self.device)
        out = []
        for start in range(0, len(s), 8192):
            z = torch.randn(len(s[start:start + 8192]), self.noise_dim, generator=gen, device=self.device)
            out.append(self.G(s[start:start + 8192], z, c[start:start + 8192])[0].cpu())
        return torch.cat(out).numpy()

    def audit(self):
        return {"refiner": self.name, "epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr,
                "noise_dim": self.noise_dim, "hidden": self.hidden, "max_shift": self.max_shift,
                "lambda_prox": self.lambda_prox, "history": self.history}
