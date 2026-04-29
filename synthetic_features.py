from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random

import networkx as nx
import torch
from torch import Tensor


@dataclass
class FeatureEngineConfig:
    mode: str
    num_nodes: int
    feature_dim: int
    noise_seed: int
    noise_std: float


@dataclass
class NodeFeatureConfig:
    mode: str
    num_nodes: int
    node_feature_dim: int
    noise_seed: int
    noise_std: float
    embedding_dim: int
    walk_length: int
    walks_per_node: int
    context_size: int
    node2vec_p: float
    node2vec_q: float
    gae_refresh_interval: int
    gae_steps: int
    gae_lr: float
    gae_max_edges: int
    gae_batch_size: int


class SyntheticFeatureEngine:
    """Strict edge-message feature generator.

    Supported modes:
    - full
    - unweighted_ones
    - gaussian_noise
    - temporal_delta
    """

    def __init__(self, config: FeatureEngineConfig) -> None:
        self.cfg = config
        self._proj_delta = self._make_projection(1, config.feature_dim, config.noise_seed + 101)
        self.reset()

    @staticmethod
    def _make_projection(in_dim: int, out_dim: int, seed: int) -> Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        proj = torch.randn((in_dim, out_dim), generator=gen, dtype=torch.float32)
        return proj / math.sqrt(float(in_dim))

    def reset(self) -> None:
        self._edge_last_time: dict[tuple[int, int], int] = {}

    def event_message(
        self,
        *,
        event_index: int,
        src: int,
        dst: int,
        current_time: int,
        base_msg: Tensor,
    ) -> Tensor:
        mode = self.cfg.mode

        if mode == "full":
            return base_msg

        if mode == "unweighted_ones":
            return torch.ones((1, self.cfg.feature_dim), dtype=torch.float32, device=base_msg.device)

        if mode == "gaussian_noise":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self.cfg.noise_seed) + int(event_index))
            noise = torch.randn((1, self.cfg.feature_dim), generator=gen, dtype=torch.float32)
            return (noise * float(self.cfg.noise_std)).to(base_msg.device)

        if mode == "temporal_delta":
            last_time = self._edge_last_time.get((src, dst))
            delta = 0.0 if last_time is None else max(0.0, float(current_time - last_time))
            raw = torch.tensor([[delta]], dtype=torch.float32, device=base_msg.device)
            return raw @ self._proj_delta.to(base_msg.device)

        raise ValueError(f"Unsupported feature mode: {mode}")

    def update_state(self, *, src: int, dst: int, current_time: int) -> None:
        self._edge_last_time[(src, dst)] = int(current_time)


class NodeFeatureGenerator:
    """Strict node initialization feature generator.

    Supported modes:
    - gaussian_noise
    - snapshot_pagerank
    - snapshot_node2vec
    - snapshot_deepwalk
    - snapshot_gae
    """

    def __init__(self, config: NodeFeatureConfig) -> None:
        self.cfg = config
        self._proj_pr = self._make_projection(1, config.node_feature_dim, config.noise_seed + 1103)
        self._proj_embed = self._make_projection(
            config.embedding_dim, config.node_feature_dim, config.noise_seed + 1707
        )
        self.reset()

    @staticmethod
    def _make_projection(in_dim: int, out_dim: int, seed: int) -> Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        proj = torch.randn((in_dim, out_dim), generator=gen, dtype=torch.float32)
        return proj / math.sqrt(float(in_dim))

    def reset(self) -> None:
        self._graph = nx.DiGraph()
        self._node_adj: dict[int, set[int]] = defaultdict(set)
        self._pagerank = torch.zeros(self.cfg.num_nodes, dtype=torch.float32)
        self._snapshot_embed = torch.zeros((self.cfg.num_nodes, self.cfg.embedding_dim), dtype=torch.float32)
        self._cached_timestamp: int | None = None
        self._cached_features = torch.zeros((self.cfg.num_nodes, self.cfg.node_feature_dim), dtype=torch.float32)
        self._updates_since_refresh = 0

    @staticmethod
    def _normalize_rows(x: Tensor) -> Tensor:
        norms = x.norm(p=2, dim=1, keepdim=True)
        return x / (norms + 1e-8)

    def _compute_pagerank(self) -> None:
        if self._graph.number_of_nodes() == 0:
            self._pagerank.zero_()
            return

        pr = nx.pagerank(self._graph, alpha=0.85, weight="weight")
        pr_vec = torch.zeros(self.cfg.num_nodes, dtype=torch.float32)
        for node, value in pr.items():
            if 0 <= int(node) < self.cfg.num_nodes:
                pr_vec[int(node)] = float(value)
        self._pagerank = pr_vec

    @staticmethod
    def _choice_with_weights(candidates: list[int], weights: list[float], rng: random.Random) -> int:
        total = sum(weights)
        if total <= 0:
            return candidates[rng.randrange(len(candidates))]

        r = rng.random() * total
        csum = 0.0
        for node, w in zip(candidates, weights):
            csum += w
            if csum >= r:
                return node
        return candidates[-1]

    def _generate_walks(self, *, current_time: int, p: float, q: float) -> list[list[int]]:
        rng = random.Random(self.cfg.noise_seed + current_time)
        nodes = [n for n in self._node_adj.keys() if self._node_adj[n]]
        if not nodes:
            return []

        walk_length = max(2, int(self.cfg.walk_length))
        walks_per_node = max(1, int(self.cfg.walks_per_node))
        walks: list[list[int]] = []

        for start in nodes:
            for _ in range(walks_per_node):
                walk = [start]
                prev: int | None = None
                cur = start

                for _step in range(walk_length - 1):
                    neighbors = list(self._node_adj[cur])
                    if not neighbors:
                        break

                    weights: list[float] = []
                    for nbr in neighbors:
                        if prev is None:
                            w = 1.0
                        elif nbr == prev:
                            w = 1.0 / max(float(p), 1e-6)
                        elif nbr in self._node_adj.get(prev, set()):
                            w = 1.0
                        else:
                            w = 1.0 / max(float(q), 1e-6)
                        weights.append(w)

                    nxt = self._choice_with_weights(neighbors, weights, rng)
                    walk.append(nxt)
                    prev, cur = cur, nxt

                walks.append(walk)

        return walks

    def _walks_to_embedding(self, walks: list[list[int]]) -> Tensor:
        embed = torch.zeros((self.cfg.num_nodes, self.cfg.embedding_dim), dtype=torch.float32)
        if not walks:
            return embed

        ctx = max(1, int(self.cfg.context_size))
        for walk in walks:
            n = len(walk)
            for i, center in enumerate(walk):
                lo = max(0, i - ctx)
                hi = min(n, i + ctx + 1)
                for j in range(lo, hi):
                    if i == j:
                        continue
                    context = walk[j]
                    slot = hash((context, self.cfg.noise_seed)) % self.cfg.embedding_dim
                    sign = 1.0 if (hash((center, context, self.cfg.noise_seed)) & 1) == 0 else -1.0
                    embed[center, slot] += sign

        return self._normalize_rows(embed)

    def _compute_snapshot_gae(self, *, current_time: int) -> Tensor:
        edges = list(self._graph.edges())
        if not edges:
            return torch.zeros((self.cfg.num_nodes, self.cfg.embedding_dim), dtype=torch.float32)

        max_edges = int(self.cfg.gae_max_edges)
        if max_edges > 0 and len(edges) > max_edges:
            step = max(1, len(edges) // max_edges)
            edges = edges[::step][:max_edges]

        edge_src = torch.tensor([int(src) for src, _ in edges], dtype=torch.long)
        edge_dst = torch.tensor([int(dst) for _, dst in edges], dtype=torch.long)
        num_edges = int(edge_src.numel())
        if num_edges == 0:
            return torch.zeros((self.cfg.num_nodes, self.cfg.embedding_dim), dtype=torch.float32)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(self.cfg.noise_seed) + int(current_time) + 2903)

        z = torch.randn(
            (self.cfg.num_nodes, self.cfg.embedding_dim),
            generator=gen,
            dtype=torch.float32,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([z], lr=float(self.cfg.gae_lr))

        steps = max(1, int(self.cfg.gae_steps))
        batch_size = max(1, min(int(self.cfg.gae_batch_size), num_edges))

        with torch.enable_grad():
            for _ in range(steps):
                optimizer.zero_grad()

                perm = torch.randperm(num_edges, generator=gen)[:batch_size]
                pos_src = edge_src[perm]
                pos_dst = edge_dst[perm]

                neg_src = pos_src
                neg_dst = torch.randint(0, self.cfg.num_nodes, (batch_size,), generator=gen, dtype=torch.long)

                pos_score = (z[pos_src] * z[pos_dst]).sum(dim=1)
                neg_score = (z[neg_src] * z[neg_dst]).sum(dim=1)

                pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pos_score, torch.ones_like(pos_score)
                )
                neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    neg_score, torch.zeros_like(neg_score)
                )
                reg_loss = 1e-5 * (z.pow(2).mean())
                loss = pos_loss + neg_loss + reg_loss

                loss.backward()
                optimizer.step()

        with torch.no_grad():
            return self._normalize_rows(z.detach())

    def node_features_for_time(self, *, current_time: int) -> Tensor:
        mode = self.cfg.mode
        if mode == "none":
            return self._cached_features

        if mode == "snapshot_gae" and self._cached_timestamp is not None:
            refresh_interval = max(1, int(self.cfg.gae_refresh_interval))
            if self._updates_since_refresh < refresh_interval:
                return self._cached_features

        if self._cached_timestamp == current_time:
            return self._cached_features

        if mode == "gaussian_noise":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self.cfg.noise_seed) + int(current_time))
            self._cached_features = torch.randn(
                (self.cfg.num_nodes, self.cfg.node_feature_dim),
                generator=gen,
                dtype=torch.float32,
            ) * float(self.cfg.noise_std)
        elif mode == "snapshot_pagerank":
            self._compute_pagerank()
            self._cached_features = self._pagerank.unsqueeze(1) @ self._proj_pr
        elif mode == "snapshot_node2vec":
            walks = self._generate_walks(current_time=current_time, p=self.cfg.node2vec_p, q=self.cfg.node2vec_q)
            self._snapshot_embed = self._walks_to_embedding(walks)
            self._cached_features = self._snapshot_embed @ self._proj_embed
        elif mode == "snapshot_deepwalk":
            walks = self._generate_walks(current_time=current_time, p=1.0, q=1.0)
            self._snapshot_embed = self._walks_to_embedding(walks)
            self._cached_features = self._snapshot_embed @ self._proj_embed
        elif mode == "snapshot_gae":
            self._snapshot_embed = self._compute_snapshot_gae(current_time=current_time)
            self._cached_features = self._snapshot_embed @ self._proj_embed
        else:
            raise ValueError(f"Unsupported node feature mode: {mode}")

        self._cached_timestamp = current_time
        self._updates_since_refresh = 0
        return self._cached_features

    def update_state(self, *, src: int, dst: int, current_time: int) -> None:
        self._graph.add_edge(src, dst, weight=1.0)
        self._node_adj[src].add(dst)
        self._node_adj[dst].add(src)
        self._updates_since_refresh += 1
        if self.cfg.mode == "snapshot_gae":
            refresh_interval = max(1, int(self.cfg.gae_refresh_interval))
            if self._updates_since_refresh >= refresh_interval:
                self._cached_timestamp = None
        else:
            self._cached_timestamp = None
