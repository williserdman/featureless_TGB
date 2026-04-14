from __future__ import annotations

from collections import defaultdict
import math
from dataclasses import dataclass
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
    temporal_ema_alpha: float
    pagerank_interval: int
    recency_tau: float
    embedding_dim: int
    walk_length: int
    walks_per_node: int
    context_size: int
    node2vec_p: float
    node2vec_q: float


@dataclass
class NodeFeatureConfig:
    mode: str
    num_nodes: int
    node_feature_dim: int
    noise_seed: int
    noise_std: float
    temporal_ema_alpha: float
    pagerank_interval: int
    recency_tau: float
    embedding_dim: int
    walk_length: int
    walks_per_node: int
    context_size: int
    node2vec_p: float
    node2vec_q: float


class SyntheticFeatureEngine:
    """Causal synthetic feature engine.

    For non-full modes, message features are generated strictly from historical
    edges observed so far in the current replay (train->val->test within an epoch).
    """

    def __init__(self, config: FeatureEngineConfig) -> None:
        self.cfg = config

        self._proj_heur = self._make_projection(9, config.feature_dim, config.noise_seed + 101)
        self._proj_pr = self._make_projection(3, config.feature_dim, config.noise_seed + 303)
        self._proj_embed = self._make_projection(
            config.embedding_dim, config.feature_dim, config.noise_seed + 707
        )
        self.reset()

    @staticmethod
    def _make_projection(in_dim: int, out_dim: int, seed: int) -> Tensor:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        proj = torch.randn((in_dim, out_dim), generator=gen, dtype=torch.float32)
        proj = proj / math.sqrt(float(in_dim))
        return proj

    def reset(self) -> None:
        self._events_seen = 0
        self._degrees = torch.zeros(self.cfg.num_nodes, dtype=torch.float32)
        self._ema = torch.zeros(self.cfg.num_nodes, dtype=torch.float32)

        self._graph = nx.DiGraph()
        self._pagerank = torch.zeros(self.cfg.num_nodes, dtype=torch.float32)
        self._events_since_pr = 0
        self._last_timestamp: int | None = None

        self._edge_last_time: dict[tuple[int, int], int] = {}
        self._adj_unweighted: dict[int, set[int]] = defaultdict(set)
        self._adj_weighted: dict[int, dict[int, float]] = defaultdict(dict)
        self._snapshot_embed = torch.zeros(
            (self.cfg.num_nodes, self.cfg.embedding_dim), dtype=torch.float32
        )

    def _compute_pagerank_unweighted(self) -> None:
        if self._graph.number_of_nodes() == 0:
            self._pagerank.zero_()
            return

        pr = nx.pagerank(self._graph, alpha=0.85, weight="weight")
        pr_vec = torch.zeros(self.cfg.num_nodes, dtype=torch.float32)
        for node, value in pr.items():
            if 0 <= int(node) < self.cfg.num_nodes:
                pr_vec[int(node)] = float(value)
        self._pagerank = pr_vec

    def _compute_pagerank_recency(self, current_time: int) -> None:
        weighted = nx.DiGraph()
        if not self._edge_last_time:
            self._pagerank.zero_()
            return

        tau = max(self.cfg.recency_tau, 1e-6)
        for (u, v), t_last in self._edge_last_time.items():
            delta = max(0.0, float(current_time - t_last))
            w = math.exp(-delta / tau)
            weighted.add_edge(u, v, weight=w)

        if weighted.number_of_nodes() == 0:
            self._pagerank.zero_()
            return

        pr = nx.pagerank(weighted, alpha=0.85, weight="weight")
        pr_vec = torch.zeros(self.cfg.num_nodes, dtype=torch.float32)
        for node, value in pr.items():
            if 0 <= int(node) < self.cfg.num_nodes:
                pr_vec[int(node)] = float(value)
        self._pagerank = pr_vec

    @staticmethod
    def _normalize_rows(x: Tensor) -> Tensor:
        norms = x.norm(p=2, dim=1, keepdim=True)
        return x / (norms + 1e-8)

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

    def _generate_walks(
        self,
        *,
        current_time: int,
        weighted: bool,
        p: float,
        q: float,
    ) -> list[list[int]]:
        rng = random.Random(self.cfg.noise_seed + current_time)
        walks: list[list[int]] = []

        nodes = [n for n in self._adj_unweighted.keys() if self._adj_unweighted[n]]
        if not nodes:
            return walks

        walk_length = max(2, int(self.cfg.walk_length))
        walks_per_node = max(1, int(self.cfg.walks_per_node))

        for start in nodes:
            for _ in range(walks_per_node):
                walk = [start]
                prev: int | None = None
                cur = start

                for _step in range(walk_length - 1):
                    neighbors = list(self._adj_unweighted[cur])
                    if not neighbors:
                        break

                    weights: list[float] = []
                    for nbr in neighbors:
                        if prev is None:
                            w = 1.0
                        else:
                            if nbr == prev:
                                w = 1.0 / max(float(p), 1e-6)
                            elif nbr in self._adj_unweighted.get(prev, set()):
                                w = 1.0
                            else:
                                w = 1.0 / max(float(q), 1e-6)

                        if weighted:
                            ew = self._adj_weighted.get(cur, {}).get(nbr, 0.0)
                            w *= max(ew, 1e-8)

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

    def _compute_snapshot_embedding(self, *, current_time: int, mode: str) -> None:
        if mode == "snapshot_node2vec":
            walks = self._generate_walks(
                current_time=current_time,
                weighted=False,
                p=self.cfg.node2vec_p,
                q=self.cfg.node2vec_q,
            )
            self._snapshot_embed = self._walks_to_embedding(walks)
            return

        if mode == "snapshot_deepwalk":
            walks = self._generate_walks(
                current_time=current_time,
                weighted=False,
                p=1.0,
                q=1.0,
            )
            self._snapshot_embed = self._walks_to_embedding(walks)
            return

        if mode == "recency_node2vec":
            tau = max(self.cfg.recency_tau, 1e-6)
            self._adj_weighted = defaultdict(dict)
            for (u, v), t_last in self._edge_last_time.items():
                delta = max(0.0, float(current_time - t_last))
                self._adj_weighted[u][v] = math.exp(-delta / tau)
                self._adj_weighted[v][u] = self._adj_weighted[u][v]

            # Use recency-weighted edges in transition probabilities.
            walks = self._generate_walks(
                current_time=current_time,
                weighted=True,
                p=self.cfg.node2vec_p,
                q=self.cfg.node2vec_q,
            )
            self._snapshot_embed = self._walks_to_embedding(walks)
            return

        raise ValueError(f"Unsupported snapshot embedding mode: {mode}")

    def _prepare_pagerank_if_needed(self, mode: str, current_time: int) -> None:
        if mode == "temporal_heuristics":
            self._events_since_pr += 1
            if self._events_since_pr >= max(1, self.cfg.pagerank_interval):
                self._compute_pagerank_unweighted()
                self._events_since_pr = 0
            return

        if mode == "snapshot_pagerank":
            if self._last_timestamp is None:
                self._last_timestamp = current_time
            elif current_time != self._last_timestamp:
                self._compute_pagerank_unweighted()
                self._last_timestamp = current_time
            return

        if mode == "recency_pagerank":
            if self._last_timestamp is None:
                self._last_timestamp = current_time
            elif current_time != self._last_timestamp:
                self._compute_pagerank_recency(current_time)
                self._last_timestamp = current_time
            return

        if mode in {"snapshot_node2vec", "snapshot_deepwalk", "recency_node2vec"}:
            if self._last_timestamp is None:
                self._last_timestamp = current_time
            elif current_time != self._last_timestamp:
                self._compute_snapshot_embedding(current_time=current_time, mode=mode)
                self._last_timestamp = current_time

    def _edge_msg_from_heuristics(self, src: int, dst: int) -> Tensor:
        deg_s = self._degrees[src].item()
        deg_d = self._degrees[dst].item()
        ema_s = self._ema[src].item()
        ema_d = self._ema[dst].item()
        pr_s = self._pagerank[src].item()
        pr_d = self._pagerank[dst].item()

        raw = torch.tensor(
            [
                deg_s,
                deg_d,
                ema_s,
                ema_d,
                pr_s,
                pr_d,
                abs(deg_s - deg_d),
                abs(ema_s - ema_d),
                abs(pr_s - pr_d),
            ],
            dtype=torch.float32,
        )
        return (raw @ self._proj_heur).unsqueeze(0)

    def _edge_msg_from_pagerank(self, src: int, dst: int) -> Tensor:
        pr_s = self._pagerank[src].item()
        pr_d = self._pagerank[dst].item()
        raw = torch.tensor([pr_s, pr_d, abs(pr_s - pr_d)], dtype=torch.float32)
        return (raw @ self._proj_pr).unsqueeze(0)

    def _edge_msg_from_snapshot_embedding(self, src: int, dst: int) -> Tensor:
        emb = 0.5 * (self._snapshot_embed[src] + self._snapshot_embed[dst])
        return (emb @ self._proj_embed).unsqueeze(0)

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

        if mode == "gaussian_noise":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self.cfg.noise_seed) + int(event_index))
            noise = torch.randn((1, self.cfg.feature_dim), generator=gen, dtype=torch.float32)
            return noise * float(self.cfg.noise_std)

        if mode not in {"temporal_heuristics", "snapshot_pagerank", "recency_pagerank"}:
            if mode not in {"snapshot_node2vec", "snapshot_deepwalk", "recency_node2vec"}:
                raise ValueError(f"Unsupported feature mode: {mode}")

        self._prepare_pagerank_if_needed(mode, current_time)

        if mode == "temporal_heuristics":
            return self._edge_msg_from_heuristics(src, dst)

        if mode in {"snapshot_node2vec", "snapshot_deepwalk", "recency_node2vec"}:
            return self._edge_msg_from_snapshot_embedding(src, dst)

        return self._edge_msg_from_pagerank(src, dst)

    def update_state(self, *, src: int, dst: int, current_time: int) -> None:
        self._degrees[src] += 1.0
        self._degrees[dst] += 1.0

        alpha = float(self.cfg.temporal_ema_alpha)
        self._ema[src] = (1.0 - alpha) * self._ema[src] + alpha
        self._ema[dst] = (1.0 - alpha) * self._ema[dst] + alpha

        if self._graph.has_edge(src, dst):
            self._graph[src][dst]["weight"] += 1.0
        else:
            self._graph.add_edge(src, dst, weight=1.0)

        self._adj_unweighted[src].add(dst)
        self._adj_unweighted[dst].add(src)

        self._edge_last_time[(src, dst)] = int(current_time)

        # Rebuild local recency-weighted adjacency for walk transitions.
        tau = max(self.cfg.recency_tau, 1e-6)
        for u, v in ((src, dst), (dst, src)):
            last = self._edge_last_time.get((u, v), current_time)
            delta = max(0.0, float(current_time - last))
            w = math.exp(-delta / tau)
            self._adj_weighted[u][v] = w

        self._events_seen += 1


class NodeFeatureGenerator:
    """Causal node feature generator used for node initialization injection.

    Features at time t are computed from history observed strictly before t.
    """

    def __init__(self, config: NodeFeatureConfig) -> None:
        self.cfg = config
        self._proj_heur = self._make_projection(5, config.node_feature_dim, config.noise_seed + 1001)
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
        proj = proj / math.sqrt(float(in_dim))
        return proj

    def reset(self) -> None:
        self._degrees = torch.zeros(self.cfg.num_nodes, dtype=torch.float32)
        self._ema = torch.zeros(self.cfg.num_nodes, dtype=torch.float32)
        self._pagerank = torch.zeros(self.cfg.num_nodes, dtype=torch.float32)
        self._snapshot_embed = torch.zeros(
            (self.cfg.num_nodes, self.cfg.embedding_dim), dtype=torch.float32
        )

        self._graph = nx.DiGraph()
        self._events_since_pr = 0
        self._last_timestamp: int | None = None
        self._edge_last_time: dict[tuple[int, int], int] = {}
        self._adj_unweighted: dict[int, set[int]] = defaultdict(set)
        self._adj_weighted: dict[int, dict[int, float]] = defaultdict(dict)

        self._cached_timestamp: int | None = None
        self._cached_features = torch.zeros(
            (self.cfg.num_nodes, self.cfg.node_feature_dim), dtype=torch.float32
        )

    @staticmethod
    def _normalize_rows(x: Tensor) -> Tensor:
        norms = x.norm(p=2, dim=1, keepdim=True)
        return x / (norms + 1e-8)

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

    def _compute_pagerank_unweighted(self) -> None:
        if self._graph.number_of_nodes() == 0:
            self._pagerank.zero_()
            return
        pr = nx.pagerank(self._graph, alpha=0.85, weight="weight")
        pr_vec = torch.zeros(self.cfg.num_nodes, dtype=torch.float32)
        for node, value in pr.items():
            if 0 <= int(node) < self.cfg.num_nodes:
                pr_vec[int(node)] = float(value)
        self._pagerank = pr_vec

    def _compute_pagerank_recency(self, current_time: int) -> None:
        weighted = nx.DiGraph()
        if not self._edge_last_time:
            self._pagerank.zero_()
            return

        tau = max(self.cfg.recency_tau, 1e-6)
        for (u, v), t_last in self._edge_last_time.items():
            delta = max(0.0, float(current_time - t_last))
            w = math.exp(-delta / tau)
            weighted.add_edge(u, v, weight=w)

        if weighted.number_of_nodes() == 0:
            self._pagerank.zero_()
            return

        pr = nx.pagerank(weighted, alpha=0.85, weight="weight")
        pr_vec = torch.zeros(self.cfg.num_nodes, dtype=torch.float32)
        for node, value in pr.items():
            if 0 <= int(node) < self.cfg.num_nodes:
                pr_vec[int(node)] = float(value)
        self._pagerank = pr_vec

    def _generate_walks(
        self,
        *,
        current_time: int,
        weighted: bool,
        p: float,
        q: float,
    ) -> list[list[int]]:
        rng = random.Random(self.cfg.noise_seed + current_time)
        walks: list[list[int]] = []

        nodes = [n for n in self._adj_unweighted.keys() if self._adj_unweighted[n]]
        if not nodes:
            return walks

        walk_length = max(2, int(self.cfg.walk_length))
        walks_per_node = max(1, int(self.cfg.walks_per_node))

        for start in nodes:
            for _ in range(walks_per_node):
                walk = [start]
                prev: int | None = None
                cur = start

                for _step in range(walk_length - 1):
                    neighbors = list(self._adj_unweighted[cur])
                    if not neighbors:
                        break

                    weights: list[float] = []
                    for nbr in neighbors:
                        if prev is None:
                            w = 1.0
                        else:
                            if nbr == prev:
                                w = 1.0 / max(float(p), 1e-6)
                            elif nbr in self._adj_unweighted.get(prev, set()):
                                w = 1.0
                            else:
                                w = 1.0 / max(float(q), 1e-6)

                        if weighted:
                            ew = self._adj_weighted.get(cur, {}).get(nbr, 0.0)
                            w *= max(ew, 1e-8)

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

    def _compute_snapshot_embedding(self, *, current_time: int, mode: str) -> None:
        if mode == "snapshot_node2vec":
            walks = self._generate_walks(
                current_time=current_time,
                weighted=False,
                p=self.cfg.node2vec_p,
                q=self.cfg.node2vec_q,
            )
            self._snapshot_embed = self._walks_to_embedding(walks)
            return

        if mode == "snapshot_deepwalk":
            walks = self._generate_walks(
                current_time=current_time,
                weighted=False,
                p=1.0,
                q=1.0,
            )
            self._snapshot_embed = self._walks_to_embedding(walks)
            return

        if mode == "recency_node2vec":
            tau = max(self.cfg.recency_tau, 1e-6)
            self._adj_weighted = defaultdict(dict)
            for (u, v), t_last in self._edge_last_time.items():
                delta = max(0.0, float(current_time - t_last))
                self._adj_weighted[u][v] = math.exp(-delta / tau)
                self._adj_weighted[v][u] = self._adj_weighted[u][v]

            walks = self._generate_walks(
                current_time=current_time,
                weighted=True,
                p=self.cfg.node2vec_p,
                q=self.cfg.node2vec_q,
            )
            self._snapshot_embed = self._walks_to_embedding(walks)
            return

        raise ValueError(f"Unsupported snapshot embedding mode: {mode}")

    def _prepare_if_needed(self, mode: str, current_time: int) -> None:
        if mode == "temporal_heuristics":
            self._events_since_pr += 1
            if self._events_since_pr >= max(1, self.cfg.pagerank_interval):
                self._compute_pagerank_unweighted()
                self._events_since_pr = 0
            return

        if mode == "snapshot_pagerank":
            if self._last_timestamp is None:
                self._last_timestamp = current_time
            elif current_time != self._last_timestamp:
                self._compute_pagerank_unweighted()
                self._last_timestamp = current_time
            return

        if mode == "recency_pagerank":
            if self._last_timestamp is None:
                self._last_timestamp = current_time
            elif current_time != self._last_timestamp:
                self._compute_pagerank_recency(current_time)
                self._last_timestamp = current_time
            return

        if mode in {"snapshot_node2vec", "snapshot_deepwalk", "recency_node2vec"}:
            if self._last_timestamp is None:
                self._last_timestamp = current_time
            elif current_time != self._last_timestamp:
                self._compute_snapshot_embedding(current_time=current_time, mode=mode)
                self._last_timestamp = current_time

    def node_features_for_time(self, *, current_time: int) -> Tensor:
        mode = self.cfg.mode
        if mode == "none":
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
            self._cached_timestamp = current_time
            return self._cached_features

        self._prepare_if_needed(mode, current_time)

        if mode == "temporal_heuristics":
            raw = torch.stack(
                [
                    self._degrees,
                    self._ema,
                    self._pagerank,
                    torch.log1p(self._degrees),
                    torch.sqrt(torch.clamp(self._ema, min=0.0)),
                ],
                dim=1,
            )
            self._cached_features = raw @ self._proj_heur
        elif mode in {"snapshot_pagerank", "recency_pagerank"}:
            self._cached_features = self._pagerank.unsqueeze(1) @ self._proj_pr
        elif mode in {"snapshot_node2vec", "snapshot_deepwalk", "recency_node2vec"}:
            self._cached_features = self._snapshot_embed @ self._proj_embed
        else:
            raise ValueError(f"Unsupported node feature mode: {mode}")

        self._cached_timestamp = current_time
        return self._cached_features

    def update_state(self, *, src: int, dst: int, current_time: int) -> None:
        self._degrees[src] += 1.0
        self._degrees[dst] += 1.0

        alpha = float(self.cfg.temporal_ema_alpha)
        self._ema[src] = (1.0 - alpha) * self._ema[src] + alpha
        self._ema[dst] = (1.0 - alpha) * self._ema[dst] + alpha

        if self._graph.has_edge(src, dst):
            self._graph[src][dst]["weight"] += 1.0
        else:
            self._graph.add_edge(src, dst, weight=1.0)

        self._adj_unweighted[src].add(dst)
        self._adj_unweighted[dst].add(src)

        self._edge_last_time[(src, dst)] = int(current_time)

        tau = max(self.cfg.recency_tau, 1e-6)
        for u, v in ((src, dst), (dst, src)):
            last = self._edge_last_time.get((u, v), current_time)
            delta = max(0.0, float(current_time - last))
            w = math.exp(-delta / tau)
            self._adj_weighted[u][v] = w

        self._cached_timestamp = None
