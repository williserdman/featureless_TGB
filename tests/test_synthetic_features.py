from __future__ import annotations

import unittest

import torch

from synthetic_features import (
    FeatureEngineConfig,
    NodeFeatureConfig,
    NodeFeatureGenerator,
    SyntheticFeatureEngine,
)


class TestSyntheticFeatureEngine(unittest.TestCase):
    def test_gaussian_noise_is_deterministic_by_event_index(self) -> None:
        cfg = FeatureEngineConfig(
            mode="gaussian_noise",
            num_nodes=10,
            feature_dim=8,
            noise_seed=123,
            noise_std=1.0,
            temporal_ema_alpha=0.1,
            pagerank_interval=10,
            recency_tau=1000.0,
            embedding_dim=8,
            walk_length=6,
            walks_per_node=2,
            context_size=2,
            node2vec_p=1.0,
            node2vec_q=1.0,
        )
        engine = SyntheticFeatureEngine(cfg)

        base_msg = torch.zeros((1, 4), dtype=torch.float32)
        m1 = engine.event_message(
            event_index=7,
            src=1,
            dst=2,
            current_time=1,
            base_msg=base_msg,
        )
        engine.reset()
        m2 = engine.event_message(
            event_index=7,
            src=1,
            dst=2,
            current_time=1,
            base_msg=base_msg,
        )
        self.assertTrue(torch.allclose(m1, m2))

    def test_temporal_heuristics_is_causal(self) -> None:
        cfg = FeatureEngineConfig(
            mode="temporal_heuristics",
            num_nodes=10,
            feature_dim=16,
            noise_seed=5,
            noise_std=1.0,
            temporal_ema_alpha=0.2,
            pagerank_interval=1,
            recency_tau=1000.0,
            embedding_dim=8,
            walk_length=6,
            walks_per_node=2,
            context_size=2,
            node2vec_p=1.0,
            node2vec_q=1.0,
        )
        engine = SyntheticFeatureEngine(cfg)

        base_msg = torch.zeros((1, 4), dtype=torch.float32)

        before_first = engine.event_message(
            event_index=0,
            src=3,
            dst=4,
            current_time=10,
            base_msg=base_msg,
        )
        engine.update_state(src=3, dst=4, current_time=10)

        before_second = engine.event_message(
            event_index=1,
            src=3,
            dst=4,
            current_time=11,
            base_msg=base_msg,
        )

        self.assertEqual(before_first.shape, (1, 16))
        self.assertEqual(before_second.shape, (1, 16))
        self.assertFalse(
            torch.allclose(before_first, before_second),
            "Second event features should change after first event state update.",
        )

    def test_snapshot_node2vec_is_causal_and_fixed_dim(self) -> None:
        cfg = FeatureEngineConfig(
            mode="snapshot_node2vec",
            num_nodes=12,
            feature_dim=10,
            noise_seed=77,
            noise_std=1.0,
            temporal_ema_alpha=0.1,
            pagerank_interval=10,
            recency_tau=1000.0,
            embedding_dim=6,
            walk_length=6,
            walks_per_node=2,
            context_size=2,
            node2vec_p=0.5,
            node2vec_q=2.0,
        )
        engine = SyntheticFeatureEngine(cfg)

        base_msg = torch.zeros((1, 4), dtype=torch.float32)

        # First timestamp has no history: should be deterministic zero-history embedding projection.
        f_t10 = engine.event_message(
            event_index=0,
            src=1,
            dst=2,
            current_time=10,
            base_msg=base_msg,
        )
        self.assertEqual(f_t10.shape, (1, 10))

        # Update history with an event at t=10, then query t=11.
        engine.update_state(src=1, dst=2, current_time=10)
        f_t11 = engine.event_message(
            event_index=1,
            src=1,
            dst=2,
            current_time=11,
            base_msg=base_msg,
        )
        self.assertEqual(f_t11.shape, (1, 10))
        self.assertFalse(torch.allclose(f_t10, f_t11))

    def test_recency_node2vec_is_deterministic(self) -> None:
        cfg = FeatureEngineConfig(
            mode="recency_node2vec",
            num_nodes=12,
            feature_dim=10,
            noise_seed=99,
            noise_std=1.0,
            temporal_ema_alpha=0.1,
            pagerank_interval=10,
            recency_tau=50.0,
            embedding_dim=6,
            walk_length=6,
            walks_per_node=2,
            context_size=2,
            node2vec_p=1.0,
            node2vec_q=1.0,
        )
        base_msg = torch.zeros((1, 4), dtype=torch.float32)

        e1 = SyntheticFeatureEngine(cfg)
        e1.update_state(src=1, dst=2, current_time=10)
        e1.update_state(src=2, dst=3, current_time=10)
        out1 = e1.event_message(
            event_index=2,
            src=1,
            dst=3,
            current_time=11,
            base_msg=base_msg,
        )

        e2 = SyntheticFeatureEngine(cfg)
        e2.update_state(src=1, dst=2, current_time=10)
        e2.update_state(src=2, dst=3, current_time=10)
        out2 = e2.event_message(
            event_index=2,
            src=1,
            dst=3,
            current_time=11,
            base_msg=base_msg,
        )
        self.assertTrue(torch.allclose(out1, out2))

    def test_node_feature_generator_is_causal(self) -> None:
        cfg = NodeFeatureConfig(
            mode="temporal_heuristics",
            num_nodes=8,
            node_feature_dim=12,
            noise_seed=31,
            noise_std=1.0,
            temporal_ema_alpha=0.2,
            pagerank_interval=1,
            recency_tau=100.0,
            embedding_dim=6,
            walk_length=6,
            walks_per_node=2,
            context_size=2,
            node2vec_p=1.0,
            node2vec_q=1.0,
        )
        gen = NodeFeatureGenerator(cfg)

        f0 = gen.node_features_for_time(current_time=10)
        self.assertEqual(f0.shape, (8, 12))

        gen.update_state(src=1, dst=2, current_time=10)
        f1 = gen.node_features_for_time(current_time=11)
        self.assertFalse(torch.allclose(f0, f1))

    def test_node_gaussian_features_deterministic_per_time(self) -> None:
        cfg = NodeFeatureConfig(
            mode="gaussian_noise",
            num_nodes=8,
            node_feature_dim=12,
            noise_seed=71,
            noise_std=0.5,
            temporal_ema_alpha=0.2,
            pagerank_interval=1,
            recency_tau=100.0,
            embedding_dim=6,
            walk_length=6,
            walks_per_node=2,
            context_size=2,
            node2vec_p=1.0,
            node2vec_q=1.0,
        )
        g1 = NodeFeatureGenerator(cfg)
        g2 = NodeFeatureGenerator(cfg)

        f1 = g1.node_features_for_time(current_time=123)
        f2 = g2.node_features_for_time(current_time=123)
        self.assertTrue(torch.allclose(f1, f2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
